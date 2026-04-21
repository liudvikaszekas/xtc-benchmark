#!/usr/bin/env python3
"""
Clean and Refine Relations Script

Filters relations based on scores and VLM validation.
Includes exclusive object uniqueness filtering.
"""

import argparse
import json
import pickle
import warnings
import asyncio
import base64
import io
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

EXCLUSIVE_OBJECT_PREDICATES = {
    "carrying", "holding", "wearing", "eating", "drinking", "riding", "driving"
}

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as exc:
    VLLM_AVAILABLE = False
    VLLM_IMPORT_ERROR = exc

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



class RelationValidation(BaseModel):
    answer: str = Field(..., pattern="^(Yes|No)$")


def annotate_image_with_labels(full_image: Image.Image, 
                               subj_bbox: List[float], obj_bbox: List[float]) -> Image.Image:
    img = full_image.copy()
    draw = ImageDraw.Draw(img)
            
    def draw_bbox(bbox, color):
        x1, y1, x2, y2 = map(int, bbox)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

    draw_bbox(subj_bbox, "red")
    draw_bbox(obj_bbox, "blue")
    return img


def build_relation_prompt_text(subject_label: str, object_label: str, predicate: str) -> Tuple[str, str]:
    system_prompt = (
        "Verify the relationship between the object in the RED bounding box (Subject) and the BLUE bounding box (Object). "
        "Return JSON with key 'answer' set to 'Yes' or 'No' only."
    )
    user_text = (
        f"Subject (RED): {subject_label}\n"
        f"Object (BLUE): {object_label}\n"
        f"Relation: {subject_label} {predicate} {object_label}?"
    )
    return system_prompt, user_text


def build_lenient_prompt_text(subject_label: str, object_label: str, predicate: str) -> Tuple[str, str]:
    system_prompt = (
        "Evaluate the relationship between the object in the RED bounding box (Subject) and the BLUE bounding box (Object). "
        "If the relationship is plausible or clearly true, return 'Yes'. Only return 'No' if the relationship is impossible. "
        "Return JSON with key 'answer' set to 'Yes' or 'No' only."
    )
    user_text = (
        f"Subject (RED): {subject_label}\n"
        f"Object (BLUE): {object_label}\n"
        f"Relation: {subject_label} {predicate} {object_label}?"
    )
    return system_prompt, user_text


class RelationValidator:
    def validate_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class VLLMRelationValidator(RelationValidator):
    def __init__(self, model_name: str, num_gpus: int, max_tokens: int, temperature: float, use_flexible_spatial_prompt: bool = False):
        if not VLLM_AVAILABLE:
            detail = f" ({VLLM_IMPORT_ERROR})" if 'VLLM_IMPORT_ERROR' in globals() else ""
            raise ImportError(f"VLLM not installed or failed to import{detail}.")
        
        self.use_flexible_spatial_prompt = use_flexible_spatial_prompt
        
        print(f"Initializing VLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95,
            max_model_len=8192,
            limit_mm_per_prompt={"image": 5},
            max_num_seqs=5,
            dtype="auto",
            enforce_eager=True,
        )
        sampling_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "skip_special_tokens": True,
        }
        if GUIDED_DECODING_AVAILABLE and GuidedDecodingParams is not None:
            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=RelationValidation.model_json_schema())
        elif STRUCTURED_OUTPUTS_AVAILABLE and StructuredOutputsParams is not None:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=RelationValidation.model_json_schema(),
                disable_additional_properties=True,
            )
        else:
            print("Warning: No schema-guided decoding API available in this vLLM version; output may be unconstrained.")

        self.sampling_params = SamplingParams(**sampling_kwargs)

    def validate_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inputs = []
        for item in batch_data:
            is_strict = item['predicate'].lower() in EXCLUSIVE_OBJECT_PREDICATES
            if self.use_flexible_spatial_prompt and not is_strict:
                sys_txt, usr_txt = build_lenient_prompt_text(item['subject_label'], item['object_label'], item['predicate'])
            else:
                sys_txt, usr_txt = build_relation_prompt_text(item['subject_label'], item['object_label'], item['predicate'])
            prompt = (
                f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{usr_txt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": [item['image']]}})
            
        if not inputs: return []
        
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            try:
                results.append(json.loads(text))
            except:
                results.append({"answer": "Yes" if text.lower().startswith("yes") else "No"})
        return results




def process_scene_entry(
    entry: Dict, psg_meta: Dict, img_dir: Path, validator: RelationValidator,
    no_rel_thresh: float, pred_thresh: float, batch_size: int, use_flexible_spatial_prompt: bool
) -> Optional[Dict[str, Any]]:
    
    img_id = int(entry['img_id'])
    
    # 1. Resolve Metadata
    # 1. Resolve Metadata
    meta_entry = next((d for d in psg_meta['data'] if int(d['image_id']) == img_id), None)
    if not meta_entry:
        print(f"Warning: No metadata for image {img_id}")
        return None
        
    file_name = meta_entry['file_name']
    annotations = meta_entry.get('segments_info') or meta_entry.get('annotations', [])

    # 2. Load Image
    img_path = img_dir / file_name
    if not img_path.exists():
        img_path = img_dir / Path(file_name).name
        if not img_path.exists():
            return None
    
    try:
        full_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

    # 3. Parse Boxes
    thing_classes = psg_meta['thing_classes']
    stuff_classes = psg_meta['stuff_classes']
    pred_classes = psg_meta['predicate_classes']
    label_names = thing_classes + stuff_classes
    
    boxes_json = []
    for idx, (bbox, label_idx) in enumerate(zip(entry["bboxes"], entry["box_label"])):
        x1, y1, x2, y2 = map(int, bbox)
        seg_id = annotations[idx].get('id') if idx < len(annotations) else None
        boxes_json.append({
            "index": idx,
            "label": label_names[int(label_idx)],
            "bbox_xyxy": [x1, y1, x2, y2],
            "id": seg_id
        })

    # 4. Filter Candidates based on scores
    relations_to_validate = []
    for i, (sbj_idx, obj_idx) in enumerate(entry["pairs"]):
        sbj_idx, obj_idx = int(sbj_idx), int(obj_idx)
        scores = entry["rel_scores"][i]
        no_rel = float(scores[0])
        
        if no_rel >= no_rel_thresh: continue
        
        candidates = []
        for p_idx, score in enumerate(scores[1:]): # predicates start from index 1
            if score >= pred_thresh:
                candidates.append({"predicate": pred_classes[p_idx], "predicate_score": float(score)})
        
        if candidates:
            relations_to_validate.append({
                "subject_index": sbj_idx,
                "subject_label": boxes_json[sbj_idx]['label'],
                "object_index": obj_idx,
                "object_label": boxes_json[obj_idx]['label'],
                "subject_bbox": boxes_json[sbj_idx]['bbox_xyxy'],
                "object_bbox": boxes_json[obj_idx]['bbox_xyxy'],
                "no_relation_score": no_rel,
                "candidates": candidates
            })

    # 5. Prepare Batch for VLM
    batch_input = []
    val_map = [] # (rel_idx, cand_idx)
    
    for r_idx, rel in enumerate(relations_to_validate):
        img_crop = annotate_image_with_labels(full_image, rel['subject_bbox'], rel['object_bbox'])
        for c_idx, cand in enumerate(rel['candidates']):
            batch_input.append({
                "image": img_crop,
                "subject_label": rel['subject_label'],
                "object_label": rel['object_label'],
                "predicate": cand['predicate']
            })
            val_map.append((r_idx, c_idx))

    # 6. Validate
    validated_mask = [False] * len(batch_input)
    for i in range(0, len(batch_input), batch_size):
        try:
            results = validator.validate_batch(batch_input[i:i + batch_size])
            for j, res in enumerate(results):
                if res.get('answer', '').lower() == 'yes':
                    validated_mask[i+j] = True
        except Exception as e:
            print(f"Batch validation error: {e}")

    # 7. Construct Result
    rel_candidates_status = {}
    for g_idx, (r, c) in enumerate(val_map):
        rel_candidates_status[(r, c)] = validated_mask[g_idx]

    final_relations = []
    for r_idx, rel in enumerate(relations_to_validate):
        valid_preds = []
        for c_idx, cand in enumerate(rel['candidates']):
            if rel_candidates_status.get((r_idx, c_idx), False):
                valid_preds.append(cand)
        
        if valid_preds:
            valid_preds.sort(key=lambda x: x['predicate_score'], reverse=True)
            final_relations.append({
                "subject_index": rel['subject_index'],
                "subject_label": rel['subject_label'],
                "subject_id": boxes_json[rel['subject_index']].get('id'),
                "object_index": rel['object_index'],
                "object_label": rel['object_label'],
                "object_id": boxes_json[rel['object_index']].get('id'),
                "no_relation_score": rel['no_relation_score'],
                "predicates": valid_preds
            })

    # 8. Post-Process: Exclusive Object Constraints
    original_count = sum(len(r['predicates']) for r in final_relations)
    
    # Collect exclusive candidates: object_idx -> list of (rel_idx, p_idx, score)
    # Since structure is nested: rel -> predicates list, we track indices carefully
    excl_candidates = {} 
    
    for r_idx, rel in enumerate(final_relations):
        obj_idx = rel['object_index']
        for p_idx, p in enumerate(rel['predicates']):
            if p['predicate'].lower() in EXCLUSIVE_OBJECT_PREDICATES:
                if obj_idx not in excl_candidates: excl_candidates[obj_idx] = []
                excl_candidates[obj_idx].append((r_idx, p_idx, p['predicate_score']))

    to_remove = set()
    for obj_idx, cands in excl_candidates.items():
        if len(cands) > 1:
            cands.sort(key=lambda x: x[2], reverse=True)
            for i in range(1, len(cands)): # Keep top 1
                to_remove.add((cands[i][0], cands[i][1]))

    # Apply removals
    if to_remove:
        cleaned_rels = []
        for r_idx, rel in enumerate(final_relations):
            new_preds = [p for p_idx, p in enumerate(rel['predicates']) if (r_idx, p_idx) not in to_remove]
            if new_preds:
                rel['predicates'] = new_preds
                cleaned_rels.append(rel)
        final_relations = cleaned_rels

    final_count = sum(len(r['predicates']) for r in final_relations)
    removed_count = original_count - final_count
    if removed_count > 0:
        print(f"[{img_id}] Post-process exclusive filter removed {removed_count} predicates.")

    return {
        "image_id": img_id,
        "file_name": file_name,
        "boxes": boxes_json,
        "relations": final_relations
    }


def process_pkl_file(
    pkl_path: Path, output_dir: Path, img_dir: Path,
    psg_meta_path: Path, validator: RelationValidator,
    no_rel_thresh: float, pred_thresh: float, batch_size: int,
    start_id: Optional[int], max_imgs: Optional[int], use_flexible_spatial_prompt: bool
):
    if not pkl_path.exists(): 
        raise FileNotFoundError(f"Input file not found: {pkl_path}")
    
    with open(psg_meta_path) as f: psg_meta = json.load(f)
    # Check if psg_meta has 'data' (and is not empty). If not, try loading 'custom_psg.json' from input dir
    if not psg_meta.get('data'):
        custom_psg_path = pkl_path.parent / "custom_psg.json"
        if custom_psg_path.exists():
            print(f"Loading metadata from {custom_psg_path}...")
            with open(custom_psg_path) as f:
                custom_meta = json.load(f)
                # Verify custom meta has data
                if 'data' in custom_meta:
                    # Merge or replace. Here we replace relevant fields or just use custom_meta
                    # But we might want to keep classes from psg_meta if they are the source of truth?
                    # Usually custom_psg.json has everything.
                    psg_meta = custom_meta
                else:
                    print(f"Warning: {custom_psg_path} also lacks 'data' key.")
        else:
             print(f"Warning: 'data' key missing in {psg_meta_path} and 'custom_psg.json' not found in {pkl_path.parent}")

    with open(pkl_path, 'rb') as f: data = pickle.load(f)

    # JSONL Setup
    jsonl_path = output_dir / "clean_relations.jsonl"
    try:
        from utils import load_processed_ids, append_jsonl
    except ImportError:
        sys.path.append(str(Path(__file__).parent))
        from utils import load_processed_ids, append_jsonl

    processed_ids = load_processed_ids(jsonl_path, "image_id")
    
    # Filter data based on processed_ids and start_id
    filtered_data = []
    for d in data:
        img_id = int(d['img_id'])
        if start_id and img_id < start_id: continue
        if img_id in processed_ids: continue
        filtered_data.append(d)
        
    data = filtered_data
    if max_imgs: data = data[:max_imgs]
        
    print(f"Processing {len(data)} images...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'val': 0}
    for entry in tqdm(data, desc="Clean & Refine"):
        res = process_scene_entry(
            entry, psg_meta, img_dir, validator,
            no_rel_thresh, pred_thresh, batch_size, use_flexible_spatial_prompt
        )
        if res:
            out_file = output_dir / f"scene-graph_{res['image_id']}.json"
            with open(out_file, 'w') as f:
                json.dump(res, f, indent=2)
            append_jsonl(jsonl_path, res)
            for r in res['relations']:
                stats['val'] += len(r.get('predicates', []))
    
    print(f"Processed {len(data)} images. Validated relations: {stats['val']}")
    print(f"Results saved to: {output_dir}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--psg-meta', required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')

    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--no-relation-threshold', type=float, default=0.5)
    parser.add_argument('--predicate-threshold', type=float, default=0.4)
    parser.add_argument('--start-image-id', type=int)
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--use-flexible-spatial-prompt', action='store_true', help='Use flexible spatial prompts.')

    args = parser.parse_args()
    
    validator = VLLMRelationValidator(args.model, args.num_gpus, args.max_tokens, args.temperature, use_flexible_spatial_prompt=args.use_flexible_spatial_prompt)

    process_pkl_file(
        Path(args.input), Path(args.output), Path(args.images),
        Path(args.psg_meta), validator, args.no_relation_threshold, args.predicate_threshold,
        args.batch_size, args.start_image_id, args.max_images, args.use_flexible_spatial_prompt
    )

if __name__ == '__main__':
    main()
