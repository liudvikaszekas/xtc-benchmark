#!/usr/bin/env python3
"""
Batch Generate Enhanced Prompts from Scene Graphs using LLM

This script iterates over scene graphs and generates improved captions using:
- LLM (VLLM) for object descriptions
- PSGEval for spatial relations
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_caption_from_sg_llm import (
    generate_caption_from_scene_graph_llm,
    load_scene_graph_from_file,
    prepare_object_crops_for_llm,
    submit_vllm_slurm_job,
    extract_relation_descriptions,
    compose_caption_from_components,
    load_segmentation_data
)
from PIL import Image
from collections import defaultdict
from llm_clients import VLLMClient


def process_scene_graphs_with_llm(
    scene_graphs_dir: Path,
    image_dir: Path,
    seg_dir: Path,
    output_dir: Path,
    llm_config: dict,
    debug_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    slurm_config: Optional[dict] = None,
    log_dir: Optional[Union[str, Path]] = None
):
    """
    Batch process scene graphs with LLM enhancement.
    
    Args:
        scene_graphs_dir: Directory with scene graph JSON files
        image_dir: Directory with original images
        seg_dir: Directory with segmentation masks
        output_dir: Output directory for prompts
        llm_config: LLM configuration dict
        debug_dir: Optional debug directory
        max_images: Optional limit on number of images to process
        slurm_config: Optional Slurm configuration for VLLM inference
    """
    if not scene_graphs_dir.exists():
        print(f"Error: Scene graphs directory {scene_graphs_dir} does not exist")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all scene graph files
    scene_graph_files = sorted(list(scene_graphs_dir.glob("scene-graph_*.json")))
    if not scene_graph_files:
        print(f"Warning: No scene graph files found in {scene_graphs_dir}")
        return False
    
    print(f"Found {len(scene_graph_files)} scene graph files")
    
    # Limit number of images for debugging if specified
    if max_images is not None and max_images > 0:
        original_count = len(scene_graph_files)
        scene_graph_files = scene_graph_files[:max_images]
        print(f"DEBUG: Limiting to {len(scene_graph_files)} images (out of {original_count} total)")
    
    # Initialize LLM client
    if slurm_config:
        print("Slurm outsourcing enabled: Skipping local VLLM initialization")
        # Create a dummy client object that just holds configuration
        class DummyVLLMClient:
            def __init__(self, config):
                self.model_name = config.get("model", "Qwen/Qwen2.5-VL-32B-Instruct")
                self.num_gpus = config.get("num_gpus", 4)
                self.batch_size = config.get("batch_size", 8)
                self.max_tokens = config.get("max_tokens", 100)
                self.temperature = config.get("temperature", 0.3)
        
        llm_client = DummyVLLMClient(llm_config)
    else:
        print("Initializing local LLM client...")
        llm_client = VLLMClient(
            model_name=llm_config.get("model", "Qwen/Qwen2.5-VL-32B-Instruct"),
            num_gpus=llm_config.get("num_gpus", 4),
            max_tokens=llm_config.get("max_tokens", 100),
            temperature=llm_config.get("temperature", 0.3)
        )
    
    batch_size = llm_config.get("batch_size", 8)
    
    # Process each scene graph
    prompts = []
    successful = 0
    failed = 0
    
    # Optimized processing: Massive batching for Slurm, original loop for local
    if slurm_config:
        print(f"Starting MASSIVE BATCH preparation for Slurm execution...")
        
        all_batch_data = [] 
        global_indices = [] # list of (sg_list_index, obj_index)
        prepared_scene_graphs = [] # list of dicts with preloaded data
        
        # 1. PREPARE ALL DATA LOCALLY
        for i, sg_file in enumerate(scene_graph_files):
            try:
                # Load scene graph
                scene_graph_data = load_scene_graph_from_file(sg_file)
                
                if 'boxes' not in scene_graph_data:
                    print(f"Skipping {sg_file.name}: 'boxes' key missing")
                    failed += 1
                    continue
                
                image_id = scene_graph_data.get('image_id')
                file_name = scene_graph_data.get('file_name', f"{image_id:012d}.jpg")
                
                # Find image file
                image_path = image_dir / file_name
                if not image_path.exists():
                    possible_names = [f"{image_id:012d}.jpg", f"{image_id}.jpg", f"COCO_val2017_{image_id:012d}.jpg"]
                    for name in possible_names:
                        if (image_dir / name).exists():
                            image_path = image_dir / name
                            break
                            
                if not image_path.exists():
                    print(f"Warning: Image not found for {sg_file.name}, skipping")
                    failed += 1
                    continue
                
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Load segmentation
                seg_masks = None
                if seg_dir and seg_dir.exists():
                    seg_masks, _ = load_segmentation_data(image_id, seg_dir)
                    if seg_masks is None:
                        # Only warn if strictly needed? For now we proceed without masks if missing
                        pass
                
                # Prepare crops
                batch_data, indices = prepare_object_crops_for_llm(
                    scene_graph_data, image, seg_masks, debug_dir, image_id
                )
                
                # Store globally
                all_batch_data.extend(batch_data)
                for idx in indices:
                    global_indices.append((len(prepared_scene_graphs), idx))
                
                prepared_scene_graphs.append({
                    'file': sg_file,
                    'data': scene_graph_data,
                    'image_id': image_id,
                    'file_name': file_name
                })
                
                if (i+1) % 10 == 0:
                    print(f"Prepared {i+1}/{len(scene_graph_files)} images...")
                    
            except Exception as e:
                print(f"Error preparing {sg_file.name}: {e}")
                failed += 1

        if not all_batch_data:
            print("No objects found to process!")
            return False

        # 2. RUN SINGLE SLURM JOB
        print(f"Starting Slurm inference for {len(all_batch_data)} objects...")
        
        # Use dummy indices 0..N for the job interface
        job_indices = list(range(len(all_batch_data)))
        
        # Use log dir if provided, else fallback
        slurm_log_dir = Path(log_dir) if log_dir else output_dir / "logs"
        slurm_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Slurm logs will be saved to: {slurm_log_dir}")
        
        # Submit job (image_id=999999 for temp naming)
        all_descriptions_list = submit_vllm_slurm_job(
            all_batch_data, job_indices, slurm_config, llm_config, 
            image_id=999999, log_dir=slurm_log_dir
        )
        
        # 3. ASSEMBLE RESULTS
        print("Assembling results...")
        
        # Map: sg_list_index -> {obj_index_or_tuple: description}
        # For groups, key will be (index, member_idx), for singles just index
        sg_obj_descriptions = defaultdict(dict)
        
        for k, desc in enumerate(all_descriptions_list):
            if k >= len(global_indices):
                break # Should not happen
            sg_idx, obj_idx_or_tuple = global_indices[k]
            sg_obj_descriptions[sg_idx][obj_idx_or_tuple] = desc
        
        # 4. GENERATE FINAL PROMPTS - with group handling
        for i, sg_info in enumerate(prepared_scene_graphs):
            try:
                sg_data = sg_info['data']
                raw_descs = sg_obj_descriptions[i]
                
                # Reconstruct object descriptions, combining group members
                obj_descs = {}
                for box in sg_data.get("boxes", []):
                    index = box.get("index")
                    label = box.get("label", "")
                    member_attributes = box.get("member_attributes", [])
                    is_group = member_attributes and len(member_attributes) > 1
                    
                    if is_group:
                        # Collect member descriptions
                        member_descs = []
                        for m_idx in range(1, len(member_attributes) + 1):
                            member_key = f"{index}_member_{m_idx}"
                            if member_key in raw_descs:
                                desc = raw_descs[member_key]
                                member_descs.append(desc)
                                # Store individual member description for composer
                                obj_descs[member_key] = desc
                        
                        # Compose group description
                        if member_descs:
                            plural_label = f"{label}s" if not label.endswith('s') else label
                            if len(member_descs) == 1:
                                obj_descs[index] = f"a {label}: {member_descs[0]}"
                            elif len(member_descs) == 2:
                                obj_descs[index] = f"a group of {len(member_descs)} {plural_label}: {member_descs[0]}, and {member_descs[1]}"
                            else:
                                members_str = ", ".join(member_descs[:-1]) + f", and {member_descs[-1]}"
                                obj_descs[index] = f"a group of {len(member_descs)} {plural_label}: {members_str}"
                        else:
                            # Fallback
                            plural_label = f"{label}s" if not label.endswith('s') else label
                            obj_descs[index] = f"a group of {len(member_attributes)} {plural_label}"
                    else:
                        # Single object
                        if index in raw_descs:
                            obj_descs[index] = raw_descs[index]
                        else:
                            obj_descs[index] = f"a {label}"
                
                # Relations
                rel_descs = extract_relation_descriptions(sg_data)
                
                # Compose
                caption = compose_caption_from_components(obj_descs, rel_descs, sg_data)
                
                prompt_entry = {
                    "image_id": sg_info['image_id'],
                    "file_name": sg_info['file_name'],
                    "prompt": caption,
                    "object_descriptions": obj_descs,
                    "relation_descriptions": rel_descs,
                    "source_file": sg_info['file'].name
                }
                prompts.append(prompt_entry)
                successful += 1
            except Exception as e:
                print(f"Error assembling prompt for {sg_info['file'].name}: {e}")
                failed += 1

    else:
        # --- ORIGINAL LOCAL LOOP ---
        for sg_file in scene_graph_files:
            try:
                # Load scene graph
                scene_graph_data = load_scene_graph_from_file(sg_file)
                
                # Basic validation
                if 'boxes' not in scene_graph_data:
                    print(f"Skipping {sg_file.name}: 'boxes' key missing")
                    failed += 1
                    continue
                
                image_id = scene_graph_data.get('image_id')
                file_name = scene_graph_data.get('file_name', f"{image_id:012d}.jpg")
                
                # Find image file
                image_path = image_dir / file_name
                if not image_path.exists():
                    possible_names = [
                        f"{image_id:012d}.jpg",
                        f"{image_id}.jpg",
                        f"COCO_val2017_{image_id:012d}.jpg"
                    ]
                    for name in possible_names:
                        alt_path = image_dir / name
                        if alt_path.exists():
                            image_path = alt_path
                            break
                    
                    if not image_path.exists():
                        print(f"Warning: Image not found for {sg_file.name}, using first available format")
                        image_path = None
                
                if image_path is None:
                    print(f"Skipping {sg_file.name}: Image not found")
                    failed += 1
                    continue
                
                # Generate enhanced caption
                caption, obj_descriptions, rel_descriptions = generate_caption_from_scene_graph_llm(
                    scene_graph_data,
                    image_path,
                    seg_dir,
                    llm_client,
                    batch_size,
                    debug_dir=debug_dir,
                    slurm_config=slurm_config 
                )
                
                # Store result
                prompt_entry = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "prompt": caption,
                    "object_descriptions": obj_descriptions,
                    "relation_descriptions": rel_descriptions,
                    "source_file": sg_file.name
                }
                
                prompts.append(prompt_entry)
                successful += 1
                
                if successful % 10 == 0:
                    print(f"Processed {successful}/{len(scene_graph_files)} scene graphs...")
            
            except Exception as e:
                print(f"Error processing {sg_file.name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue
    
    # Save outputs
    # 1. Structured JSON
    output_json = output_dir / 'prompts.json'
    with open(output_json, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    # 2. Simple text file
    output_txt = output_dir / 'prompts.txt'
    with open(output_txt, 'w') as f:
        for p in prompts:
            f.write(f"Image {p['image_id']}: {p['prompt']}\n\n")
    
    # 3. Detailed output with object descriptions
    output_detailed = output_dir / 'prompts_detailed.json'
    with open(output_detailed, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\n✓ Generated {successful} prompts")
    print(f"✓ Saved to: {output_json}")
    print(f"✓ Saved text version to: {output_txt}")
    print(f"✓ Saved detailed version to: {output_detailed}")
    
    if failed > 0:
        print(f"⚠ Failed to process {failed} scene graphs")
    
    return successful > 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced prompts from scene graphs using LLM"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing scene graph JSON files"
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing original images"
    )
    parser.add_argument(
        "--seg-dir",
        default=None,
        help="Directory containing segmentation masks (optional)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save generated prompts"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="VLLM model name"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for LLM inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Directory to save debug images and attribute JSONs"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for debugging, e.g., 5)"
    )
    parser.add_argument(
        "--slurm-config",
        type=str,
        default=None,
        help="Path to JSON file containing Slurm configuration for VLLM outsourcing"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save Slurm logs"
    )
    
    args = parser.parse_args()
    
    llm_config = {
        "model": args.model,
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature
    }
    
    slurm_config = None
    if args.slurm_config:
        try:
            with open(args.slurm_config, 'r') as f:
                slurm_config = json.load(f)
            print(f"Loaded Slurm config from {args.slurm_config}")
        except Exception as e:
            print(f"Error loading Slurm config file: {e}")
            sys.exit(1)
    
    success = process_scene_graphs_with_llm(
        Path(args.input_dir),
        Path(args.image_dir),
        Path(args.seg_dir) if args.seg_dir else None,
        Path(args.output_dir),
        llm_config,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        max_images=args.max_images,
        slurm_config=slurm_config,
        log_dir=args.log_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
