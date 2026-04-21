#!/usr/bin/env python3
"""
Generate object attributes using VLM with structured output and guided decoding.
Features:
1. Semantic Spotlight (Context): Blurs background while keeping the target sharp using segmentation.
2. Solid Crop (Detail): Isolates the object on grey to prevent bleeding.
3. Visual Reasoning: Forces the model to evaluate image clarity before guessing.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import contextlib
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as exc:
    VLLM_AVAILABLE = False
    VLLM_IMPORT_ERROR = exc
    LLM = Any  # type: ignore
    SamplingParams = Any  # type: ignore
    GuidedDecodingParams = None

try:
    from vllm.sampling_params import GuidedDecodingParams
    GUIDED_DECODING_AVAILABLE = True
except Exception:
    GuidedDecodingParams = None
    GUIDED_DECODING_AVAILABLE = False

try:
    from vllm.sampling_params import StructuredOutputsParams
    STRUCTURED_OUTPUTS_AVAILABLE = True
except Exception:
    StructuredOutputsParams = None
    STRUCTURED_OUTPUTS_AVAILABLE = False


# ==========================================
# 1. Schema Definition (The Safety Net)
# ==========================================

def create_dynamic_schema(attribute_types: List[str], force_all: bool = False):
    """
    Create Pydantic schema. Includes 'visual_reasoning' to force Chain-of-Thought.
    """
    fields = {}
    annotations = {}
    
    # 1. Add visual evidence field FIRST
    annotations['visual_reasoning'] = str
    fields['visual_reasoning'] = Field(
        ..., 
        description="Step-by-step analysis of image clarity. State if text/logos are legible or if the image is blurry/pixelated."
    )
    
    # 2. Add attribute fields
    for attr_type in attribute_types:
        field_name = attr_type.replace(' ', '_').replace('-', '_').lower()
        if force_all:
            annotations[field_name] = List[str]
            fields[field_name] = Field(
                ...,
                description=f"{attr_type} attributes. List visible ones or ['unknown'] if not.",
            )
        else:
            annotations[field_name] = Optional[List[str]]
            fields[field_name] = Field(
                default=None,
                description=f"{attr_type} attributes. Only include if clearly visible/applicable.", 
            )
    
    return type('DynamicObjectAttributes', (BaseModel,), {
        '__annotations__': annotations,
        **fields
    })


# ==========================================
# 2. Image Processing Utilities
# ==========================================

def upscale_if_small(img: Image.Image, min_side: int = 224, max_side: int = 768) -> Image.Image:
    """Upscale small crops so the vision encoder sees enough pixels."""
    w, h = img.size
    s = min(w, h)
    if s >= min_side:
        return img

    scale = min_side / float(s)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    max_dim = max(new_w, new_h)
    if max_dim > max_side:
        scale2 = max_side / float(max_dim)
        new_w, new_h = int(round(new_w * scale2)), int(round(new_h * scale2))

    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def create_spotlight_image(full_image: Image.Image, mask_array: np.ndarray, seg_id: int, bbox: List[float], blur_radius: int = 15) -> Image.Image:
    """
    Creates a 'Semantic Spotlight' (Portrait Mode).
    The exact shape of the object remains sharp. The background is blurred.
    Also draws a red bounding box.
    """
    # 1. Create binary mask from the segmentation (Object = White, Background = Black)
    binary_mask = (mask_array == seg_id).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask, mode='L')
    
    # 2. Dilate (expand) the mask slightly to include immediate context edges
    # This prevents the edges of the object from getting blurred
    mask_dilated = mask_img.filter(ImageFilter.MaxFilter(size=21)) 
    
    # 3. Soften the transition (Feathering)
    mask_soft = mask_dilated.filter(ImageFilter.GaussianBlur(radius=3))
    
    # 4. Create the Blurred Background
    # Radius 15 destroys text legibility but keeps shapes recognizable.
    blurred_bg = full_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 5. Composite: Paste the Sharp Image onto the Blurred BG using the Mask
    result = Image.composite(full_image, blurred_bg, mask_soft)

    # Draw red bounding box for attention
    if bbox:
        draw = ImageDraw.Draw(result)
        draw.rectangle(bbox, outline="red", width=3)

    return result


def extract_object_image(full_image: Image.Image, mask: np.ndarray, seg_id: int, bbox: List[float]) -> Image.Image:
    """
    Extract object on a SOLID GREY background for detail analysis.
    """
    binary_mask = (mask == seg_id).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask, mode='L')
    
    # Feather edges
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))

    # USE SOLID GREY BACKGROUND (127,127,127)
    bg = Image.new("RGB", full_image.size, (127, 127, 127))
    
    result = Image.composite(full_image, bg, mask_img)

    # Draw thinner bounding box
    draw = ImageDraw.Draw(result)
    draw.rectangle(bbox, outline="red", width=2)

    x1, y1, x2, y2 = map(int, bbox)
    
    # Context padding: 20%
    w_box, h_box = x2 - x1, y2 - y1
    pad_x = int(w_box * 0.2)
    pad_y = int(h_box * 0.2)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(full_image.width, x2 + pad_x)
    y2 = min(full_image.height, y2 + pad_y)

    crop = result.crop((x1, y1, x2, y2))
    crop = upscale_if_small(crop, min_side=224, max_side=1024)

    return crop


def get_bbox_from_mask(mask: np.ndarray, seg_id: int) -> Optional[List[int]]:
    rows = np.any(mask == seg_id, axis=1)
    cols = np.any(mask == seg_id, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax + 1), int(ymax + 1)]


# ==========================================
# 3. Model & Generation
# ==========================================

def initialize_vllm_model(num_gpus: int = 4, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> LLM:
    if not VLLM_AVAILABLE:
        detail = f" ({VLLM_IMPORT_ERROR})" if 'VLLM_IMPORT_ERROR' in globals() else ""
        raise ImportError(f"VLLM library not found or failed to import{detail}")

    engine_args = {
        "model": model_name,
        "limit_mm_per_prompt": {"image": 2}, 
        "dtype": "bfloat16",
        "tensor_parallel_size": num_gpus,
        "max_model_len": 8192,
        "max_num_seqs": 5,
        "enforce_eager": True,
    }
    print(f"Loading model: {model_name}")
    llm = LLM(**engine_args)
    return llm


def generate_fake_attributes(attribute_types: List[str]) -> Dict:
    return {attr_type.replace(' ', '_').replace('-', '_').lower(): ["test"] for attr_type in attribute_types}


def generate_attributes_batch(
    llm: Optional[LLM],
    image_paths: List[Tuple[str, str]],
    category_names: List[str],
    attribute_types: List[str],
    max_tokens: int = 768,
    fake: bool = False,
    is_prediction: bool = False
) -> List[Dict]:
    if fake:
        fake_attrs = generate_fake_attributes(attribute_types)
        return [fake_attrs] * len(image_paths)
    
    placeholder = "<|image_pad|>"
    dynamic_schema = create_dynamic_schema(attribute_types, force_all=is_prediction)
    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "skip_special_tokens": True,
    }
    if GUIDED_DECODING_AVAILABLE:
        sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=dynamic_schema.model_json_schema())
    elif STRUCTURED_OUTPUTS_AVAILABLE:
        sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
            json=dynamic_schema.model_json_schema(),
            disable_additional_properties=True,
        )
    else:
        print("Warning: No schema-guided decoding API available in this vLLM version; output may be unconstrained.")

    sampling_params = SamplingParams(**sampling_kwargs)
    
    attr_types_str = ", ".join(attribute_types)
    
    if is_prediction:
        instruction = "For each potential attribute, if visible, list it. If an attribute is NOT visible or NOT applicable, set the value to ['unknown']."
    else:
        instruction = "Omit the field if the attribute is not clearly visible or applicable."
    
    inputs = []
    for (spotlight_img_path, cropped_img_path), cat_name in zip(image_paths, category_names):
        try:
            spotlight_data = Image.open(spotlight_img_path)
            cropped_data = Image.open(cropped_img_path)
            
            prompt = (
                f"<|im_start|>system\n"
                "You are a visual attribute extractor. Follow this strict procedure:\n"
                "1. IMAGE ANALYSIS ROLES:\n"
                "   - Image 1 (Spotlight): Use ONLY for context (scale, orientation, scene type). Do NOT read text or attributes from here.\n"
                "   - Image 2 (Detail Crop): This is the SOURCE OF TRUTH. Use this image for all text, color, and material attributes.\n"
                "2. VISUAL REASONING:\n"
                "   - First, fill the 'visual_reasoning' field based on Image 2.\n"
                "   - Describe the clarity of Image 2. If it is pixelated/blurry, state 'Text is unreadable'.\n"
                "3. ATTRIBUTE EXTRACTION:\n"
                f"   - {instruction}\n"
                "   - Do NOT describe people or noise visible in the blurred background of Image 1.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<|vision_start|>{placeholder}<|vision_end|>"
                f"<|vision_start|>{placeholder}<|vision_end|>"
                f"Extract attributes for the target {cat_name}.\n"
                f"Potential Attributes: {attr_types_str}.\n"
                "Return a JSON object. Only include keys for attributes that are clearly visible. Omit keys for missing/unclear attributes.\n"
                "<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": [spotlight_data, cropped_data]},
                "multi_modal_uuids": {"image": [spotlight_img_path, cropped_img_path]},
            })
        except Exception as e:
            print(f"Warning: Failed to load images: {e}")
            inputs.append(None)
    
    valid_inputs = [inp for inp in inputs if inp is not None]
    valid_indices = [i for i, inp in enumerate(inputs) if inp is not None]
    
    if not valid_inputs:
        return [{}] * len(image_paths)
    
    outputs = llm.generate(valid_inputs, sampling_params=sampling_params)
    attributes_list = [{}] * len(image_paths)
    
    for idx, output in zip(valid_indices, outputs):
        try:
            text = output.outputs[0].text.strip()
            attributes_dict = json.loads(text)
            attributes_list[idx] = {k: v for k, v in attributes_dict.items() if v}
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            attributes_list[idx] = {}
    
    return attributes_list


# ==========================================
# 4. Data Loading & Main Loop
# ==========================================

def load_category_mapping(mapping_path: str) -> Dict:
    with open(mapping_path) as f:
        return json.load(f)

def load_anno_data(anno_path: str) -> Dict:
    with open(anno_path) as f:
        return json.load(f)

def load_scene_graph_indices(scene_graphs_dir: Path, category_mapping: Dict = None, anno_data: Dict = None) -> Dict:
    index_map_by_cat_bbox = {}
    index_map_by_index = {}
    
    if not scene_graphs_dir or not scene_graphs_dir.exists():
        return {'by_cat_bbox': index_map_by_cat_bbox, 'by_index': index_map_by_index}
    
    label_to_cat_id = {}
    if anno_data:
        all_categories = anno_data.get('thing_classes', []) + anno_data.get('stuff_classes', [])
        label_to_cat_id = {cat: idx for idx, cat in enumerate(all_categories)}
    
    for sg_file in scene_graphs_dir.glob("scene-graph_*.json"):
        try:
            with open(sg_file, 'r') as f:
                scene_graph = json.load(f)
            
            image_id = scene_graph.get("image_id")
            if image_id is None: continue
            
            for box in scene_graph.get("boxes", []):
                category_id = box.get("category_id")
                label = box.get("label")
                bbox_xyxy = box.get("bbox_xyxy", [])
                index = box.get("index")
                
                if index is None or not bbox_xyxy or len(bbox_xyxy) != 4: continue
                
                index_map_by_index[(image_id, index)] = True
                
                if category_id is None and label and label in label_to_cat_id:
                    category_id = label_to_cat_id[label]
                
                if category_id is not None:
                    bbox_key_normalized = tuple(int(round(x / 5)) * 5 for x in bbox_xyxy)
                    key = (image_id, category_id, bbox_key_normalized)
                    index_map_by_cat_bbox[key] = index
        except Exception as e:
            print(f"Warning: Failed to parse scene graph {sg_file.name}: {e}")
    
    return {'by_cat_bbox': index_map_by_cat_bbox, 'by_index': index_map_by_index}


def process_annotations(
    anno_data: Dict,
    category_mapping: Dict,
    img_dir: Path,
    seg_dir: Path,
    llm: Optional[LLM],
    batch_size: int = 8,
    scene_graphs_dir: Optional[Path] = None,
    fake: bool = False,
    target_image_ids: Optional[List[int]] = None,
    debug_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    is_prediction_step: bool = False
) -> List[Dict]:
    results = []
    all_categories = anno_data['thing_classes'] + anno_data['stuff_classes']
    supercategories = category_mapping['supercategories']
    cat_to_supercat = category_mapping['category_to_supercategory']
    
    scene_graph_index_map = {}
    if scene_graphs_dir:
        print(f"Loading scene graph indices from: {scene_graphs_dir}")
        index_maps = load_scene_graph_indices(scene_graphs_dir, category_mapping, anno_data)
        scene_graph_index_map = index_maps['by_cat_bbox']
    
    objects_to_process = []
    
    total_entries = len(anno_data.get('data', []))
    print(f"anno_data contains {total_entries} image entries.")
    if total_entries == 0:
        print("ERROR: anno_data['data'] is empty. Check that the segmentation step (step 1) completed successfully.")
        print(f"  anno.json keys: {list(anno_data.keys())}")
    
    skip_filter = 0
    skip_missing_img = 0
    skip_missing_seg = 0
    skip_load_error = 0
    
    print("Preparing object list...")
    for image_entry in anno_data['data']:
        image_id = image_entry['image_id']
        if target_image_ids and image_id not in target_image_ids:
            skip_filter += 1
            continue

        file_name = image_entry['file_name']
        img_path = img_dir / file_name
        seg_path = seg_dir / image_entry['pan_seg_file_name']
        
        if not img_path.exists():
            skip_missing_img += 1
            if skip_missing_img <= 5:
                print(f"  Skip (missing image): {img_path}")
            elif skip_missing_img == 6:
                print("  ... (suppressing further missing image warnings)")
            continue
        if not seg_path.exists():
            skip_missing_seg += 1
            if skip_missing_seg <= 5:
                print(f"  Skip (missing seg mask): {seg_path}")
            elif skip_missing_seg == 6:
                print("  ... (suppressing further missing seg mask warnings)")
            continue
        
        try:
            full_image = Image.open(img_path).convert("RGB")
            seg_mask_orig = np.array(Image.open(seg_path))
            if len(seg_mask_orig.shape) == 3:
                seg_mask = seg_mask_orig[:, :, 0].astype(np.int32) + seg_mask_orig[:, :, 1].astype(np.int32) * 256 + seg_mask_orig[:, :, 2].astype(np.int32) * 256 * 256
            else:
                seg_mask = seg_mask_orig.astype(np.int32)
        except Exception as e:
            skip_load_error += 1
            if skip_load_error <= 5:
                print(f"Warning: Failed to load data for image_id={image_id}: {e}")
            continue
        
        for ann_idx, annotation in enumerate(image_entry['annotations']):
            cat_id = annotation['category_id']
            cat_name = all_categories[cat_id]
            seg_id = image_entry['segments_info'][ann_idx]['id'] if ann_idx < len(image_entry['segments_info']) else (ann_idx + 1)
            
            bbox_from_mask = get_bbox_from_mask(seg_mask, seg_id)
            if bbox_from_mask:
                bbox_xyxy = bbox_from_mask
                x, y, x2, y2 = bbox_xyxy
                bbox = [x, y, x2 - x, y2 - y]
            else:
                bbox = annotation['bbox']
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    bbox_xyxy = [x, y, x + w, y + h]
                else:
                    bbox_xyxy = bbox
            
            supercat = cat_to_supercat.get(cat_name, "object")
            attr_types = supercategories.get(supercat, {"attribute_types": []}).get("attribute_types", [])
            
            bbox_key_normalized = tuple(int(round(x / 5)) * 5 for x in bbox_xyxy)
            index_key = (image_id, cat_id, bbox_key_normalized)
            box_index = scene_graph_index_map.get(index_key)
            
            objects_to_process.append({
                "image_id": image_id,
                "category_id": cat_id,
                "category_name": cat_name,
                "supercategory": supercat,
                "attribute_types": attr_types,
                "bbox": bbox,
                "bbox_xyxy": bbox_xyxy,
                "full_image": full_image,
                "seg_mask": seg_mask,
                "seg_id": seg_id,
                "index": box_index
            })
    
    print(f"\nTotal objects to process: {len(objects_to_process)}")
    print(f"  Entries in anno_data: {total_entries}")
    if skip_filter: print(f"  Skipped (target image filter): {skip_filter}")
    if skip_missing_img: print(f"  Skipped (missing image file): {skip_missing_img}")
    if skip_missing_seg: print(f"  Skipped (missing seg mask file): {skip_missing_seg}")
    if skip_load_error: print(f"  Skipped (load error): {skip_load_error}")
    if total_entries > 0 and len(objects_to_process) == 0:
        print("  WARNING: All entries were skipped. Check that img-dir and seg-dir paths are correct.")
        print(f"    img-dir: {img_dir}")
        print(f"    seg-dir: {seg_dir}")

    # JSONL Support
    jsonl_path = output_dir / "attributes.jsonl" if output_dir else Path("attributes.jsonl")
    processed_keys = set()
    try:
        from utils import append_jsonl
        if jsonl_path.exists():
            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        processed_keys.add((d['image_id'], d['seg_id']))
                    except: pass
        print(f"Resuming: Found {len(processed_keys)} processed objects.")
    except ImportError:
        sys.path.append(str(Path(__file__).parent))
        from utils import append_jsonl
        if jsonl_path.exists():
            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        processed_keys.add((d['image_id'], d['seg_id']))
                    except: pass
    
    # Filter
    objects_to_process = [obj for obj in objects_to_process if (obj['image_id'], obj['seg_id']) not in processed_keys]
    print(f"Remaining objects to process: {len(objects_to_process)}")
    
    objects_by_supercat = {}
    for obj in objects_to_process:
        supercat = obj['supercategory']
        if supercat not in objects_by_supercat:
            objects_by_supercat[supercat] = []
        objects_by_supercat[supercat].append(obj)
    
    cm = TemporaryDirectory() if debug_dir is None else contextlib.nullcontext(str(debug_dir))
    with cm as temp_dir:
        temp_path = Path(temp_dir)
        if debug_dir:
            temp_path.mkdir(parents=True, exist_ok=True)
        
        for supercat, supercat_objects in tqdm(objects_by_supercat.items(), desc="Processing"):
            for batch_start in range(0, len(supercat_objects), batch_size):
                batch_end = min(batch_start + batch_size, len(supercat_objects))
                batch = supercat_objects[batch_start:batch_end]
            
                image_paths = []
                category_names = []
                
                for i, obj in enumerate(batch):
                    try:
                        name_base = f"{obj['image_id']}_{obj['seg_id']}_{supercat}"
                        
                        spotlight_img = create_spotlight_image(
                            obj["full_image"], 
                            obj["seg_mask"], 
                            obj["seg_id"], 
                            obj["bbox_xyxy"],
                            blur_radius=5
                        )
                        spotlight_path = temp_path / f"{name_base}_spotlight.png"
                        spotlight_img.save(spotlight_path)
                        
                        obj_img = extract_object_image(
                            obj["full_image"],
                            obj["seg_mask"],
                            obj["seg_id"],
                            obj["bbox_xyxy"]
                        )
                        
                        def is_extreme_aspect_ratio(img):
                            w, h = img.size
                            return max(w, h) / max(min(w, h), 1) > 150
                            
                        if is_extreme_aspect_ratio(spotlight_img) or is_extreme_aspect_ratio(obj_img):
                            print(f"Warning: Skipping {name_base} due to extreme aspect ratio.")
                            image_paths.append((None, None))
                            category_names.append("")
                            continue
                            
                        cropped_img_path = temp_path / f"{name_base}_obj.png"
                        obj_img.save(cropped_img_path)
                        
                        image_paths.append((str(spotlight_path), str(cropped_img_path)))
                        category_names.append(obj["category_name"])
                    except Exception as e:
                        print(f"Warning: Failed to extract object: {e}")
                        image_paths.append((None, None))
                        category_names.append("")
                
                valid_paths = [(fp, cp) for fp, cp in image_paths if fp is not None]
                valid_names = [n for (fp, cp), n in zip(image_paths, category_names) if fp is not None]
                
                if valid_paths:
                    batch_attr_types = batch[0]["attribute_types"]
                    attributes_batch = generate_attributes_batch(
                        llm, valid_paths, valid_names, batch_attr_types, fake=fake, is_prediction=is_prediction_step
                    )
                    
                    attr_idx = 0
                    for i, obj in enumerate(batch):
                        if image_paths[i][0] is not None:
                            attributes = attributes_batch[attr_idx]
                            attr_idx += 1
                        else:
                            attributes = {}
                        
                        result = {
                            "image_id": obj["image_id"],
                            "category_id": obj["category_id"],
                            "category": obj["category_name"],
                            "supercategory": obj["supercategory"],
                            "bbox": obj["bbox"],
                            "seg_id": obj["seg_id"],
                            "attributes": attributes
                        }
                        if obj.get("index") is not None:
                            result["index"] = obj["index"]
                        
                        results.append(result)
                        append_jsonl(jsonl_path, result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate object attributes using VLM")
    
    parser.add_argument("--anno-json", type=str, default="../output_seg/anno.json")
    parser.add_argument("--img-dir", type=str, default="../images")
    parser.add_argument("--seg-dir", type=str, default="../output_seg")
    parser.add_argument("--mapping-json", type=str, default="./updated_category_mapping.json")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scene-graphs-dir", type=str, default=None)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-ids", type=str, default=None)
    parser.add_argument("--debug-dir", type=str, default=None)
    
    parser.add_argument("--is-prediction-step", action="store_true")
    
    args = parser.parse_args()
    
    anno_path = Path(args.anno_json)
    img_dir = Path(args.img_dir)
    seg_dir = Path(args.seg_dir)
    mapping_path = Path(args.mapping_json)
    output_dir = Path(args.output_dir)
    
    target_image_ids = None
    if args.image_ids:
        try:
            target_image_ids = [int(x.strip()) for x in args.image_ids.split(',')]
            print(f"Filtering for {len(target_image_ids)} image IDs: {target_image_ids}")
        except ValueError:
            print("Error parsing image IDs.")
            exit(1)

    print("=== VLM Attribute Generation (Hybrid Spotlight Mode) ===\n")
    print("Loading annotation data...")
    anno_data = load_anno_data(anno_path)
    print("\nLoading category mapping...")
    category_mapping = load_category_mapping(mapping_path)
    
    llm = None
    if not args.fake:
        llm = initialize_vllm_model(args.num_gpus, args.model)
    else:
        print("Using FAKE mode")
    
    print(f"\nTargeting output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Processing Annotations ===")
    scene_graphs_dir = Path(args.scene_graphs_dir) if args.scene_graphs_dir else None
    results = process_annotations(
        anno_data,
        category_mapping,
        img_dir,
        seg_dir,
        llm,
        args.batch_size,
        scene_graphs_dir=scene_graphs_dir,
        fake=args.fake,
        target_image_ids=target_image_ids,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        output_dir=output_dir,
        is_prediction_step=args.is_prediction_step
    )
    
    output_path = output_dir / "attributes.json"
    
    if target_image_ids and output_path.exists():
         print(f"Updating existing attributes file: {output_path}")
         with open(output_path, 'r') as f:
             existing_results = json.load(f)
         updated_results = [r for r in existing_results if r.get('image_id') not in target_image_ids]
         updated_results.extend(results)
         results = updated_results
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Generated attributes for {len(results)} objects")
    print(f"✓ Results saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())