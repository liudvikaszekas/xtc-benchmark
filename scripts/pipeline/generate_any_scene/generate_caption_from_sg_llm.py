#!/usr/bin/env python3
"""
Enhanced Caption Generator from Scene Graphs using LLM + PSGEval Hybrid

This module generates improved captions by:
1. Using LLM (VLLM) to create rich object descriptions based on visual features + attributes
2. Using PSGEval method for spatial relation descriptions
3. Composing the final caption from both sources

Supports both local and Slurm execution modes for VLLM inference.
"""

import json
import pickle
import subprocess
import time
import tempfile
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np

# Import PSGEval components from existing script
from gas.captions_generation.scene_graph import get_sg_desc
from gas.captions_generation.utils import convert_sg_to_json

# Import LLM clients
from llm_clients import LLMClient, VLLMClient


def generate_semantic_ids(scene_graph_data: Dict) -> Dict[int, str]:
    """
    Generate unique semantic IDs (e.g. Person_1, PersonGroup_2) for all boxes.
    Matches logic in utils_group_prompts.py.
    """
    box_semantic_ids = {}  # idx -> "Person_1"
    label_counters = {}
    
    for idx, box in enumerate(scene_graph_data.get("boxes", [])):
        raw_label = box.get("label", "object")
        member_attrs = box.get("member_attributes", [])
        is_group = member_attrs and len(member_attrs) > 1

        # Sanitize label
        base_label_key = raw_label.lower().replace(" ", "_")
        
        # Determine counter key based on group status
        if is_group:
            label_key = f"{base_label_key}_group"
        else:
            label_key = base_label_key

        label_counters[label_key] = label_counters.get(label_key, 0) + 1
        
        base_name = raw_label.replace(' ', '_').capitalize()
        if is_group:
            semantic_id = f"{base_name}Group_{label_counters[label_key]}"
        else:
            semantic_id = f"{base_name}_{label_counters[label_key]}"
            
        box_semantic_ids[idx] = semantic_id
        
    return box_semantic_ids


def load_segmentation_data(image_id: int, seg_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Load segmentation masks and annotation data for an image.
    
    Args:
        image_id: Image ID
        seg_dir: Directory containing segmentation PNG files and anno.json
        
    Returns:
        Tuple of (mask array with shape (H, W), annotation dict with segments_info), or (None, None) if not found
    """
    # Load annotation JSON to get segment info
    anno_path = seg_dir / "anno.json"
    anno_data = None
    if anno_path.exists():
        with open(anno_path, 'r') as f:
            anno_json = json.load(f)
            # Check different possible structures
            if isinstance(anno_json, dict):
                # Check if there's a 'data' key (kmax-deeplab format)
                if 'data' in anno_json and isinstance(anno_json['data'], list):
                    # List format inside 'data' key
                    for entry in anno_json['data']:
                        if entry.get("image_id") == image_id:
                            anno_data = entry
                            break
                else:
                    # Dict format: keys are string image IDs
                    anno_data = anno_json.get(str(image_id))
            elif isinstance(anno_json, list):
                # List format: find entry with matching image_id
                for entry in anno_json:
                    if entry.get("image_id") == image_id:
                        anno_data = entry
                        break
    
    # Try different possible filenames (kmax-deeplab uses seg_*.png format)
    possible_names = [
        f"seg_{image_id:012d}.png",  # kmax-deeplab format
        f"seg_{image_id}.png", # visual genome format
        f"{image_id:012d}.png",       # COCO format
        f"{image_id}.png",
        f"segmentation_{image_id}.png"
    ]
    
    for name in possible_names:
        mask_path = seg_dir / name
        if mask_path.exists():
            # Load PNG and convert to indices
            from PIL import Image as PILImage
            mask_img = PILImage.open(mask_path)
            mask = np.array(mask_img)
            
            # PNG segmentation masks use RGB encoding for segment IDs
            # Convert RGB to single integer ID
            if len(mask.shape) == 3:
                mask = mask[:, :, 0].astype(np.int32) + \
                       mask[:, :, 1].astype(np.int32) * 256 + \
                       mask[:, :, 2].astype(np.int32) * 256 * 256
            
            return mask, anno_data
    
    # Don't print warning - it's fine to work without masks
    return None, None


def find_matching_segment(bbox_xyxy: List[float], segments_info: List[Dict], mask: np.ndarray) -> Optional[int]:
    """
    Find the best matching segment ID for a given bounding box using IoU.
    
    Args:
        bbox_xyxy: Bounding box [x1, y1, x2, y2]
        segments_info: List of segment info dicts from anno.json
        mask: Segmentation mask array
        
    Returns:
        Segment ID (for indexing into the mask), or None if no good match
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox_xyxy]
    bbox_area = (x2 - x1) * (y2 - y1)
    
    if bbox_area == 0:
        return None
    
    best_iou = 0.0
    best_segment_id = None
    
    for seg_info in segments_info:
        seg_id = seg_info.get("id")
        if seg_id is None:
            continue
        
        # Calculate IoU between bbox and segment mask
        seg_mask = (mask == seg_id)
        
        # Get segment bbox from mask
        ys, xs = np.where(seg_mask)
        if len(ys) == 0:
            continue
        
        seg_x1, seg_y1 = xs.min(), ys.min()
        seg_x2, seg_y2 = xs.max() + 1, ys.max() + 1
        
        # Calculate intersection
        inter_x1 = max(x1, seg_x1)
        inter_y1 = max(y1, seg_y1)
        inter_x2 = min(x2, seg_x2)
        inter_y2 = min(y2, seg_y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue  # No intersection
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        seg_area = seg_info.get("area", (seg_x2 - seg_x1) * (seg_y2 - seg_y1))
        union_area = bbox_area + seg_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        if iou > best_iou:
            best_iou = iou
            best_segment_id = seg_id
    
    # Only return if IoU is reasonable (> 0.3)
    return best_segment_id if best_iou > 0.3 else None


def extract_object_region(
    image: Image.Image,
    bbox_xyxy: List[float],
    seg_map: Optional[np.ndarray] = None,
    segment_ids: Optional[List[int]] = None,
    exclude_seg_ids: Optional[List[int]] = None,
    debug_dir: Optional[Path] = None,
    image_id: Optional[int] = None,
    object_index: Optional[int] = None,
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
    on_missing_mask: str = "blank",  # "blank" | "raise" (I recommend blank)
) -> Image.Image:
    x1, y1, x2, y2 = [int(c) for c in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (1, 1), background_rgb)

    cropped = image.crop((x1, y1, x2, y2))

    # Safe behavior: if seg info missing, return blank crop (prevents leakage)
    if seg_map is None or not segment_ids:
        if on_missing_mask == "raise":
            raise RuntimeError(f"Missing seg_map/segment_ids for image_id={image_id} obj={object_index}")
        blank = Image.new("RGB", cropped.size, background_rgb)
        if debug_dir and image_id is not None and object_index is not None:
            blank.save(debug_dir / f"img_{image_id:012d}_obj_{object_index:03d}_SKIPPED.png")
        return blank

    seg_crop = seg_map[y1:y2, x1:x2]

    include = np.zeros(seg_crop.shape, dtype=bool)
    for sid in segment_ids:
        include |= (seg_crop == int(sid))

    if exclude_seg_ids:
        exclude = np.zeros(seg_crop.shape, dtype=bool)
        for sid in exclude_seg_ids:
            exclude |= (seg_crop == int(sid))
        include &= ~exclude

    mask_pil = Image.fromarray((include.astype(np.uint8) * 255), mode="L")
    background = Image.new("RGB", cropped.size, background_rgb)
    result = Image.composite(cropped, background, mask_pil)

    if debug_dir and image_id is not None and object_index is not None:
        result.save(debug_dir / f"img_{image_id:012d}_obj_{object_index:03d}.png")

    return result




def extract_relation_descriptions(scene_graph_data: Dict) -> List[str]:
    """
    Extract spatial relations from scene graph using PSGEval method.
    
    This creates a minimal NetworkX graph with just objects and relations
    (no attributes) to get clean relation descriptions.
    
    Args:
        scene_graph_data: Scene graph dict
        
    Returns:
        List of relation description strings
    """
    G = nx.DiGraph()
    
    # Add object nodes (without attributes)
    boxes = scene_graph_data.get("boxes", [])
    for box in boxes:
        obj_id = f"object_{box['index'] + 1}"
        G.add_node(obj_id, type="object_node", value=box["label"])
    
    if "relations" not in scene_graph_data:
        raise KeyError(
            f"Scene graph missing required 'relations' field. "
            f"Available fields: {list(scene_graph_data.keys())}"
        )
    
    relations = scene_graph_data["relations"]
    
    # ---------------------------------------------------------
    # 1. Generate Semantic IDs (Person_1, PersonGroup_2)
    # ---------------------------------------------------------
    box_semantic_ids = generate_semantic_ids(scene_graph_data)

    # ---------------------------------------------------------
    # 2. Build map: seg_id -> specific semantic identifier
    # ---------------------------------------------------------
    seg_id_to_sub_id = {}
    
    for idx in box_semantic_ids:
        semantic_id = box_semantic_ids[idx]
        box = boxes[idx]
        
        # Map group members
        member_attrs = box.get("member_attributes", [])
        if member_attrs and len(member_attrs) > 1:
            # Group with individual members
            for i, member in enumerate(member_attrs, 1):
                mid = member.get("seg_id")
                if mid is not None:
                    # e.g. PersonGroup_1_2
                    seg_id_to_sub_id[mid] = f"{semantic_id}_{i}"
        else:
            # Flat box: map its seg_ids (list or scalar) to the main ID
            seg_ids = box.get("seg_ids", [])
            if isinstance(seg_ids, list):
                for sid in seg_ids:
                    seg_id_to_sub_id[sid] = semantic_id
            elif isinstance(seg_ids, (int, str)) and seg_ids is not None:
                seg_id_to_sub_id[seg_ids] = semantic_id
    
    # ---------------------------------------------------------
    # 3. Generate Relation Strings
    # ---------------------------------------------------------
    relation_descriptions = []
    
    for rel in relations:
        source_id = f"object_{rel['subject_index'] + 1}"
        target_id = f"object_{rel['object_index'] + 1}"
        
        G.add_edge(
            source_id,
            target_id,
            type="relation_edge",
            value_type="spatial",
            value=rel["predicate"]
        )
        
        # --- Granular ID Resolution ---
        subj_idx = rel.get("subject_index")
        obj_idx = rel.get("object_index")
        
        subj_seg_id = rel.get("subject_id")  # Specific scalar ID from Step 3
        obj_seg_id = rel.get("object_id")    # Specific scalar ID from Step 3
        
        # Resolve Subject
        subj_desc = seg_id_to_sub_id.get(subj_seg_id)
        if not subj_desc:
            # Fallback 1: Try first item in subject_seg_ids list
            subj_list = rel.get("subject_seg_ids")
            if subj_list and len(subj_list) > 0:
                subj_desc = seg_id_to_sub_id.get(subj_list[0])
            
            # Fallback 2: General Group ID using box index
            if not subj_desc and subj_idx in box_semantic_ids:
                subj_desc = box_semantic_ids[subj_idx]
        
        # Resolve Object
        obj_desc = seg_id_to_sub_id.get(obj_seg_id)
        if not obj_desc:
            # Fallback 1
            obj_list = rel.get("object_seg_ids")
            if obj_list and len(obj_list) > 0:
                obj_desc = seg_id_to_sub_id.get(obj_list[0])
            
            # Fallback 2
            if not obj_desc and obj_idx in box_semantic_ids:
                obj_desc = box_semantic_ids[obj_idx]
                
        # Final relation string
        predicate = rel["predicate"]
        
        if subj_desc and obj_desc:
            relation_descriptions.append(f"{subj_desc} {predicate} {obj_desc}")
        else:
            # Last resort fallback
            subject_label = rel.get("subject_label", boxes[subj_idx]["label"])
            object_label = rel.get("object_label", boxes[obj_idx]["label"])
            relation_descriptions.append(f"{subject_label} {predicate} {object_label}")
    
    return relation_descriptions


def submit_vllm_slurm_job(
    batch_data: List[Dict[str, Any]],
    indices: List[int],
    slurm_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    image_id: Optional[int] = None,
    log_dir: Optional[Path] = None
) -> List[str]:
    """
    Submit a Slurm job for VLLM inference.
    
    1. Pickle batch data to temp file
    2. Generate Slurm script
    3. Submit job
    4. Wait for completion
    5. Load results from output pickle
    """
    # Determine directories
    if log_dir is None:
        # Fallback if no log_dir provided
        base_dir = Path.cwd()
        log_dir = base_dir / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Input pickle goes to parent directory (e.g., 8_prompts_gt/)
    # unless log_dir is just "logs" in cwd, then maybe cwd?
    # Assuming log_dir is usually output_dir/logs
    work_dir = log_dir.parent
    
    timestamp = int(time.time())
    if image_id is not None:
        filename_base = f"vllm_{image_id}_{timestamp}"
    else:
        filename_base = f"vllm_{timestamp}"
        
    input_pickle = work_dir / "input.pkl"
    # To avoid overwriting if running multiple, maybe add fallback? 
    # But user asked for specific name "input.pkl" in 8_prompts_(gt).
    # If we run multiple batches, this might conflict. 
    # For now, following user request strictly for "input.pkl".
    
    # Actually, let's keep it safe but compliant. 
    # If this is a huge batch (batch_generate_prompts_llm calls it once for all), 
    # input.pkl is fine.
    
    output_pickle = work_dir / f"output_{filename_base}.pkl"
    
    print(f"  Preparing Slurm job for {len(batch_data)} objects...")
    print(f"  Saving input to {input_pickle}")
    with open(input_pickle, 'wb') as f:
        pickle.dump({
            'batch_data': batch_data,
            'indices': indices,
            'metadata': {'image_id': image_id}
        }, f)
    
    # Generate Slurm script
    script_path = log_dir / "vllm_grounding_job.slurm"
    slurm_script = _generate_vllm_slurm_script(
        input_pickle, output_pickle, slurm_config, llm_config, log_dir
    )
    
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    print(f"  Generated Slurm script: {script_path}")
    
    # Submit job
    print(f"  Submitting Slurm job...")
    result = subprocess.run(
        ['sbatch', str(script_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Slurm submission failed: {result.stderr}")
    
    # Parse job ID
    import re
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
    
    job_id = match.group(1)
    print(f"  ✓ Submitted job {job_id}, waiting for completion...")
    
    try:
        # Wait for job completion and stream logs
        _wait_for_slurm_job(job_id, log_dir=log_dir)
        
        # Load results
        if not output_pickle.exists():
            raise RuntimeError(f"Output pickle not found: {output_pickle}")
        
        with open(output_pickle, 'rb') as f:
            results = pickle.load(f)
        
        descriptions = results['descriptions']
        print(f"  ✓ Received {len(descriptions)} descriptions from Slurm job")
        
        return descriptions
        
    finally:
        # cleanup could happen here, but maybe user wants to inspect input.pkl?
        # keeping files as requested.
        pass


def _generate_vllm_slurm_script(
    input_pickle: Path,
    output_pickle: Path,
    slurm_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    log_dir: Optional[Path] = None
) -> str:
    """Generate Slurm batch script for VLLM inference."""
    account = slurm_config.get('account', 'sci-zacharatou')
    partition = slurm_config.get('partition', 'gpu-batch')
    nodes = slurm_config.get('nodes', 1)
    constraint = slurm_config.get('constraint')
    gpus_type = slurm_config.get('gpus_type') # Default to None
    num_gpus = llm_config.get('num_gpus', 4)
    mem = slurm_config.get('mem', '100G')
    cpus = slurm_config.get('cpus', 8)
    time_limit = slurm_config.get('time', '00:30:00')
    
    model = llm_config.get('model', 'Qwen/Qwen2.5-VL-32B-Instruct')
    batch_size = llm_config.get('batch_size', 8)
    max_tokens = llm_config.get('max_tokens', 100)
    temperature = llm_config.get('temperature', 0.3)
    
    use_container = slurm_config.get('use_container', True)
    container_name = slurm_config.get('container_name', 'vllm')
    container_mounts = slurm_config.get('container_mounts', '/sc/home:/sc/home')
    
    # Build GPU request
    if gpus_type:
        gpu_request = f"--gpus={gpus_type}:{num_gpus}"
    else:
        gpu_request = f"--gpus={num_gpus}"
    
    # Get script path
    script_dir = Path(__file__).parent
    vllm_script = script_dir / 'vllm_inference_only.py'
    
    # Check log dir
    log_path = log_dir if log_dir else output_pickle.parent
    if log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    
    script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name=vllm_grounding
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH {gpu_request}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --output={log_path}/vllm_grounding_%j.out
#SBATCH --error={log_path}/vllm_grounding_%j.err
"""
    if constraint:
        script += f"#SBATCH --constraint={constraint}\n"
    
    if use_container:
        script += f"""#SBATCH --container-name={container_name}
#SBATCH --container-mounts={container_mounts}
"""
    
    script += f"""
# Run VLLM inference
python3 {vllm_script} \\
    --input-pickle {input_pickle} \\
    --output-pickle {output_pickle} \\
    --model {model} \\
    --num-gpus {num_gpus} \\
    --batch-size {batch_size} \\
    --max-tokens {max_tokens} \\
    --temperature {temperature}

exit $?
"""
    
    return script


def _wait_for_slurm_job(job_id: str, poll_interval: int = 5, max_wait: int = 7200, log_dir: Optional[Path] = None, prefix: str = "vllm_grounding"):
    """Wait for Slurm job to complete and stream logs."""
    start_time = time.time()
    log_offsets = {}
    last_status = None
    
    while time.time() - start_time < max_wait:
        # Check job status
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            capture_output=True,
            text=True
        )
        
        status = result.stdout.strip()
        
        # Tail logs if log_dir is provided
        if log_dir:
            for log_file in log_dir.glob(f"*{job_id}.*"):
                if log_file.suffix not in ('.out', '.err'):
                    continue
                path_str = str(log_file)
                if path_str not in log_offsets:
                    log_offsets[path_str] = 0
                try:
                    with open(log_file, 'rb') as f:
                        f.seek(log_offsets[path_str])
                        chunk = f.read()
                        if chunk:
                            text = chunk.decode('utf-8', errors='replace')
                            lines = text.splitlines(keepends=True)
                            for line in lines:
                                print(f"[{prefix}] {line}", end='')
                            log_offsets[path_str] = f.tell()
                except Exception:
                    pass

        if result.returncode != 0 or not status:
            # Job finished
            break
        
        if status != last_status:
            print(f"\n[{prefix}] [Job {job_id} status: {status}]")
            last_status = status
            
        time.sleep(poll_interval)
    
    print(f"\n[{prefix}] Job finished processing.")
    
    # Check exit code
    result = subprocess.run(
        ['sacct', '-j', job_id, '-n', '-X', '-o', 'ExitCode'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        exit_code_str = result.stdout.strip()
        # Sometimes sacct takes a moment to update
        if not exit_code_str:
            time.sleep(2)
            result = subprocess.run(
                ['sacct', '-j', job_id, '-n', '-X', '-o', 'ExitCode'],
                capture_output=True,
                text=True
            )
            exit_code_str = result.stdout.strip()
            
        if ':' in exit_code_str:
            exit_code = exit_code_str.split(':')[0]
        else:
            exit_code = exit_code_str
        
        if exit_code != '0':
            raise RuntimeError(f"Slurm job {job_id} failed with exit code: {exit_code}")


def generate_llm_object_descriptions(
    scene_graph_data: Dict,
    image: Image.Image,
    seg_masks: Optional[np.ndarray],
    seg_anno: Optional[Dict],  # kept for compatibility; not required
    llm_client: LLMClient,
    batch_size: int = 8,
    debug_dir: Optional[Path] = None,
    image_id: Optional[int] = None,
    slurm_config: Optional[Dict[str, Any]] = None
) -> Dict[int, str]:
    """
    Generate LLM descriptions for objects.
    
    Supports two modes:
    - Local: Uses llm_client directly (default)
    - Slurm: Submits VLLM-only job to Slurm (when slurm_config provided)
    
    Args:
        scene_graph_data: Scene graph dict
        image: PIL Image
        seg_masks: Segmentation mask array
        seg_anno: Annotation data (optional)
        llm_client: LLM client (used for local mode or config)
        batch_size: Batch size
        debug_dir: Debug directory
        image_id: Image ID
        slurm_config: Optional Slurm config dict. If provided, uses Slurm mode.
        
    Returns:
        Dict mapping object index to description string
    """

def prepare_object_crops_for_llm(
    scene_graph_data: Dict,
    image: Image.Image,
    seg_masks: Optional[np.ndarray],
    debug_dir: Optional[Path] = None,
    image_id: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Prepare object crops and metadata for LLM processing.
    """
    boxes = scene_graph_data.get("boxes", [])
    if not boxes:
        return [], []

    batch_data = []
    indices = []

    for box in boxes:
        index = box.get("index")
        label = box.get("label", "")
        bbox_xyxy = box.get("bbox_xyxy", [])
        
        # Merge individual member attributes for groups to give LLM full context
        attributes = box.get("attributes", {})
        
        member_attrs = box.get("member_attributes", [])
        member_attributes = member_attrs  # Store for later use
        is_group = member_attrs and len(member_attrs) > 1
        
        if is_group:
            # It's a group: Flatten member attributes into a readable format for the LLM prompt
            # e.g. "member_1": {"clothing": "red shirt"}, "member_2": ...
            group_context = {}
            for i, member in enumerate(member_attrs, 1):
                 m_attrs = member.get("attributes", {})
                 if m_attrs:
                     group_context[f"member_{i}"] = m_attrs
            
            if group_context:
                # Add this rich context to the attributes dict passed to LLM
                attributes["group_members_details"] = group_context

        if index is None or not bbox_xyxy:
            continue

        # For grouped objects, process each member individually
        if is_group:
            # Process each member separately
            for member_idx, member in enumerate(member_attributes, 1):
                member_seg_id = member.get("seg_id")
                member_attrs_dict = member.get("attributes", {})
                
                if member_seg_id is None:
                    continue
                
                # Use only this member's segment ID
                segment_ids = [int(member_seg_id)]
                
                # Exclude ALL other segment IDs (including other members of this group)
                exclude_ids = []
                for other_box in boxes:
                    for sid in (other_box.get("seg_ids") or []):
                        sid_int = int(sid)
                        if sid_int != member_seg_id:  # Exclude everything except this member
                            exclude_ids.append(sid_int)
                exclude_ids = exclude_ids if exclude_ids else None
                
                # Extract individual member region
                object_image = extract_object_region(
                    image=image,
                    bbox_xyxy=bbox_xyxy,  # Use overall group bbox (will be cropped to segment)
                    seg_map=seg_masks,
                    segment_ids=segment_ids,
                    exclude_seg_ids=exclude_ids,
                    debug_dir=debug_dir,
                    image_id=image_id,
                    object_index=f"{index}_member_{member_idx}",
                    on_missing_mask="blank",
                )
                
                batch_data.append({
                    "label": label,
                    "attributes": member_attrs_dict,  # Use member-specific attributes
                    "object_image": object_image,
                    "is_group": False,  # Each member is described individually
                    "member_count": 1,
                    "member_attributes": None,
                    "parent_index": index,  # Track which group this belongs to
                    "member_index": member_idx
                })
                indices.append(f"{index}_member_{member_idx}")  # Track as string key consistent with composer

        # Regular single object processing
        # Use SG seg_ids directly
        segment_ids = [int(s) for s in (box.get("seg_ids") or [])]

        # Exclude ALL other objects' segment IDs (not just stuff)
        # This prevents overlapping objects from appearing in this object's image
        exclude_ids = []
        for other_box in boxes:
            if other_box.get("index") != index:
                for sid in (other_box.get("seg_ids") or []):
                    exclude_ids.append(int(sid))
        exclude_ids = exclude_ids if exclude_ids else None

        object_image = extract_object_region(
            image=image,
            bbox_xyxy=bbox_xyxy,
            seg_map=seg_masks,
            segment_ids=segment_ids,
            exclude_seg_ids=exclude_ids,
            debug_dir=debug_dir,
            image_id=image_id,
            object_index=index,
            on_missing_mask="blank",
        )

        batch_data.append({
            "label": label,
            "attributes": attributes,
            "object_image": object_image,
            "is_group": is_group,
            "member_count": len(member_attributes) if is_group else 1,
            "member_attributes": member_attributes if is_group else None
        })
        indices.append(index)

        
    return batch_data, indices


def generate_llm_object_descriptions(
    scene_graph_data: Dict,
    image: Image.Image,
    seg_masks: Optional[np.ndarray],
    seg_anno: Optional[Dict],  # kept for compatibility; not required
    llm_client: LLMClient,
    batch_size: int = 8,
    debug_dir: Optional[Path] = None,
    image_id: Optional[int] = None,
    slurm_config: Optional[Dict[str, Any]] = None
) -> Dict[int, str]:
    """
    Generate LLM descriptions for objects.
    
    Supports two modes:
    - Local: Uses llm_client directly (default)
    - Slurm: Submits VLLM-only job to Slurm (when slurm_config provided)
    """
    
    # Use helper to prepare data only
    batch_data, indices = prepare_object_crops_for_llm(
        scene_graph_data, image, seg_masks, debug_dir, image_id
    )
    
    if not batch_data:
        return {}

    # DECISION POINT: Use Slurm or local execution?
    if slurm_config and slurm_config.get('use_slurm', False):
        # SLURM MODE: Submit VLLM-only job
        print(f"Using Slurm mode for VLLM inference ({len(batch_data)} objects/members)")
        
        # Extract LLM config from client or use defaults
        llm_config = {
            'model': getattr(llm_client, 'model_name', 'Qwen/Qwen2.5-VL-32B-Instruct'),
            'num_gpus': slurm_config.get('num_gpus', 4),
            'batch_size': batch_size,
            'max_tokens': getattr(llm_client, 'max_tokens', 100),
            'temperature': getattr(llm_client, 'temperature', 0.3)
        }
        
        # Submit job and get results
        all_descriptions = submit_vllm_slurm_job(
            batch_data, indices, slurm_config, llm_config, image_id
        )
        
        # Map back to indices (can be int or tuple for group members)
        raw_descriptions = {}
        for idx, desc in zip(indices, all_descriptions):
            raw_descriptions[idx] = desc
    else:
        # LOCAL MODE: Use llm_client directly (original behavior)
        print(f"Using local mode for VLLM inference ({len(batch_data)} objects/members)")
        raw_descriptions = {}
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i+batch_size]
            batch_indices = indices[i:i+batch_size]
            batch_descriptions = llm_client.batch_generate_object_descriptions(batch)
            for idx, desc in zip(batch_indices, batch_descriptions):
                raw_descriptions[idx] = desc

    # Reconstruct grouped objects by combining member descriptions
    descriptions: Dict[Any, str] = {}
    for box in scene_graph_data.get("boxes", []):
        index = box.get("index")
        label = box.get("label", "")
        member_attributes = box.get("member_attributes", [])
        is_group = member_attributes and len(member_attributes) > 1
        
        if is_group:
            # Collect member descriptions
            member_descs = []
            for i in range(1, len(member_attributes) + 1):
                member_key = f"{index}_member_{i}"
                if member_key in raw_descriptions:
                    desc = raw_descriptions[member_key]
                    member_descs.append(desc)
                    # ALSO store individual member descriptions in the main dict 
                    # so compose_caption_from_components/prompts_detailed.json see it!
                    descriptions[member_key] = desc
            
            # Compose group description
            if member_descs:
                plural_label = f"{label}s" if not label.endswith('s') else label
                if len(member_descs) == 1:
                    descriptions[index] = f"a {label}: {member_descs[0]}"
                elif len(member_descs) == 2:
                    descriptions[index] = f"a group of {len(member_descs)} {plural_label}: {member_descs[0]}, and {member_descs[1]}"
                else:
                    members_str = ", ".join(member_descs[:-1]) + f", and {member_descs[-1]}"
                    descriptions[index] = f"a group of {len(member_descs)} {plural_label}: {members_str}"
            else:
                # Fallback
                descriptions[index] = f"a group of {len(member_attributes)} {plural_label}"
        else:
            # Single object - use description directly
            if index in raw_descriptions:
                descriptions[index] = raw_descriptions[index]
            else:
                descriptions[index] = f"a {label}"
    
    return descriptions


def format_member_attributes(attributes: Dict[str, Any]) -> str:
    """Helper to convert attributes dict to string (simplified version of GroupPromptGenerator)."""
    parts = []
    
    # Order matters
    attr_order = ["color", "size", "material", "pattern", "clothing_type", "clothing_color", "action"]
    
    for key in attr_order:
        if key in attributes:
            val = attributes[key]
            if isinstance(val, list): val = " and ".join(val)
            if val:
                if "color" in key:
                    parts.append(f"in {val}")
                elif "clothing" in key:
                    parts.append(f"wearing {val}")
                elif "material" in key:
                    parts.append(f"made of {val}")
                elif "action" in key:
                    parts.append(f"that is {val}")
                else:
                    parts.append(f"with {val} {key.replace('_', ' ')}")

    # Add remaining
    for key, val in attributes.items():
        if key not in attr_order and val:
             if isinstance(val, list): val = " and ".join(val)
             parts.append(f"with {key.replace('_', ' ')}: {val}")
             
    return " ".join(parts)


def compose_caption_from_components(
    object_descriptions: Dict[Any, str],
    relation_descriptions: List[str],
    scene_graph_data: Dict
) -> str:
    """
    Compose final caption from object descriptions and relations.
    
    Now uses LLM-generated object descriptions as the primary content,
    enhanced with spatial relations.
    
    Args:
        object_descriptions: Dict of {index: description} (index can be int or string for members)
        relation_descriptions: List of relation strings
        scene_graph_data: Original scene graph (for context)
        
    Returns:
        Composed caption string
    """
    boxes = scene_graph_data.get("boxes", [])
    
    # Generate Semantic IDs to prefix objects
    box_semantic_ids = generate_semantic_ids(scene_graph_data)
    
    # Strategy: Use LLM descriptions for main objects, connect with key relations
    caption_parts = []
    
    # Get main objects with LLM descriptions (include all objects)
    main_objects = []
    for box in boxes:
        idx = box["index"]
        
        # Get Semantic ID
        semantic_id = box_semantic_ids.get(idx, "")
        
        # Determine prefix/suffix
        if semantic_id:
             prefix = f"{semantic_id} ("
             suffix = ")"
        else:
             prefix = ""
             suffix = ""

        # Check for members to append structured list
        member_attrs = box.get("member_attributes", [])
        member_lines = []
        if member_attrs and len(member_attrs) > 1:
            for i, member in enumerate(member_attrs, 1):
                # Try to get LLM description for this member
                member_key = f"{idx}_member_{i}"
                if member_key in object_descriptions:
                    m_desc = object_descriptions[member_key]
                else:
                    # Fallback to formatting attributes
                    m_attr = member.get("attributes", {})
                    m_desc = format_member_attributes(m_attr)
                
                # Form: 1) person (PersonGroup_1_1) wearing ...
                m_semantic_id = f"{semantic_id}_{i}"
                label = box['label']
                member_lines.append(f"{i}) {label} ({m_semantic_id}) {m_desc}")
            
        # Construct full description
        if member_lines:
             # Case: Group with members - strictly list members, no group description
             num_members = len(member_lines)
             plural_label = f"{box['label']}s" if not box['label'].endswith('s') else box['label']
             group_intro = f"{semantic_id} (a group of {num_members} {plural_label}, where the members are:\n  " if semantic_id else f"a group of {num_members} {plural_label}, where the members are:\n  "
             full_desc = group_intro + "\n  ".join(member_lines)
        else:
             # Case: Simple object or group without details - use group description
            if idx in object_descriptions:
                 desc = object_descriptions[idx]
            else:
                 desc = f"a {box['label']}"
            full_desc = f"{prefix}{desc}"
        
        full_desc += suffix
        main_objects.append(full_desc)
    
    # If we have objects, build caption  
    if main_objects:
        # Start with object descriptions
        if len(main_objects) == 1:
            base_caption = f"an image showing {main_objects[0]}"
        elif len(main_objects) == 2:
            base_caption = f"an image showing {main_objects[0]} and {main_objects[1]}"
        else:
            base_caption = f"an image showing {', '.join(main_objects[:-1])}, and {main_objects[-1]}"
        
        # Add spatial relations at the end if available
        if relation_descriptions and len(relation_descriptions) > 0:
            # Include all relations to give full spatial context
            relations_to_use = relation_descriptions
            if len(relations_to_use) == 1:
                relations_str = relations_to_use[0]
            elif len(relations_to_use) == 2:
                relations_str = f"{relations_to_use[0]}, and {relations_to_use[1]}"
            else:
                relations_str = ', '.join(relations_to_use[:-1]) + f", and {relations_to_use[-1]}"
            caption = f"{base_caption}, where {relations_str}"
        else:
            caption = base_caption
    else:
        # Fallback: use relations only if no object descriptions
        if relation_descriptions:
            parts = relation_descriptions
            if len(parts) == 1:
                caption = f"an image showing {parts[0]}"
            elif len(parts) == 2:
                caption = f"an image showing {parts[0]}, and {parts[1]}"
            else:
                caption = f"an image showing {', '.join(parts[:-1])}, and {parts[-1]}"
        else:
            # Ultimate fallback
            caption = "an image showing various objects"
    
    return caption


def generate_caption_from_scene_graph_llm(
    scene_graph_data: Dict,
    image_path: Path,
    seg_dir: Optional[Path],
    llm_client: LLMClient,
    batch_size: int = 8,
    debug_dir: Optional[Path] = None,
    slurm_config: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[int, str], List[str]]:
    """
    Generate enhanced caption using LLM + PSGEval hybrid approach.
    
    Workflow:
    1. Load image and segmentation masks
    2. Generate LLM descriptions for each object (with visual grounding)
    3. Extract spatial relations using PSGEval
    4. Compose final caption
    
    Supports both local and Slurm execution modes for VLLM inference.
    
    Args:
        scene_graph_data: Scene graph dict
        image_path: Path to original image
        seg_dir: Directory with segmentation masks (optional)
        llm_client: LLM client instance
        batch_size: Batch size for LLM inference
        debug_dir: Debug directory for object images
        slurm_config: Optional Slurm config dict for VLLM inference
        
    Returns:
        Tuple of (caption, object_descriptions_dict, relation_descriptions_list)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Load segmentation masks and annotation data if available
    seg_masks = None
    seg_anno = None
  
    if seg_dir and seg_dir.exists():
        image_id = scene_graph_data.get("image_id")
        seg_masks, seg_anno = load_segmentation_data(image_id, seg_dir)

        if seg_masks is None:
            raise RuntimeError(f"Missing seg map for image_id={image_id} in {seg_dir}")
    else:
        raise RuntimeError("seg_dir not provided / does not exist")
    
    # Create debug directory if enabled
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode enabled: saving object images and attributes to {debug_dir}")
    
    # Generate LLM object descriptions
    image_id = scene_graph_data.get("image_id")
    print(f"Generating LLM descriptions for {len(scene_graph_data.get('boxes', []))} objects...")
    object_descriptions = generate_llm_object_descriptions(
        scene_graph_data,
        image,
        seg_masks,
        seg_anno,
        llm_client,
        batch_size,
        debug_dir=debug_dir,
        image_id=image_id,
        slurm_config=slurm_config  # Pass through Slurm config
    )

    # --- CLEANUP: Remove parent group description if members are strictly listed ---
    boxes = scene_graph_data.get("boxes", [])
    for box in boxes:
        idx = box.get("index")
        member_attrs = box.get("member_attributes", [])
        
        # Check if we have member descriptions for this group
        has_members = False
        if member_attrs and len(member_attrs) > 1:
            for i in range(1, len(member_attrs) + 1):
                member_key = f"{idx}_member_{i}"
                if member_key in object_descriptions:
                    has_members = True
                    break
        
        # If members are present, remove the redundant parent description
        if has_members:
            if idx in object_descriptions:
                del object_descriptions[idx]
            elif str(idx) in object_descriptions:
                del object_descriptions[str(idx)]
    # -------------------------------------------------------------------------------
    
    # Extract relation descriptions
    relation_descriptions = extract_relation_descriptions(scene_graph_data)
    
    # Compose final caption
    caption = compose_caption_from_components(
        object_descriptions,
        relation_descriptions,
        scene_graph_data
    )
    
    return caption, object_descriptions, relation_descriptions


def load_scene_graph_from_file(filepath: Path) -> Dict:
    """Load scene graph from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test with example scene graph
    test_sg_path = Path("test_scene_graph.json")
    
    if test_sg_path.exists():
        print("Testing LLM-enhanced caption generation...")
        
        # Initialize VLLM client
        llm_client = VLLMClient(num_gpus=1)
        
        # Load scene graph
        scene_graph = load_scene_graph_from_file(test_sg_path)
        
        # Generate caption
        image_path = Path(scene_graph.get("file_name", "test.jpg"))
        caption, obj_descs, rel_descs = generate_caption_from_scene_graph_llm(
            scene_graph,
            image_path,
            None,
            llm_client
        )
        
        print("\n" + "="*60)
        print("Enhanced Caption:")
        print("="*60)
        print(caption)
        print("\n" + "="*60)
        print("Object Descriptions:")
        print("="*60)
        for idx, desc in obj_descs.items():
            print(f"  Object {idx}: {desc}")
        print("\n" + "="*60)
        print("Relations:")
        print("="*60)
        for rel in rel_descs:
            print(f"  - {rel}")
    else:
        print("No test scene graph found. Skipping test.")
