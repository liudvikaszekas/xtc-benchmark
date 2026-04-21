"""
Enhanced script to generate scene graphs WITH prediction masks using HiLo.

This script:
1. Loads HiLo model
2. Loads PSG ground truth annotations
3. Runs inference on specified COCO images
4. Saves both predicted and ground truth scene graphs in JSON format
5. Saves prediction masks separately for visualization
6. Format is compatible with the graph matching pipeline
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle

# Fix for older torch versions (HiLo merge code needs this)
import torch
if not hasattr(torch, 'asarray'):
    torch.asarray = torch.as_tensor

# Add HiLo to path
sys.path.insert(0, '/home/mpws25/HiLo')

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector
import time
import traceback


# ============ MASK MERGE LOGIC FROM hilo_merge_nodes.py ============

def calculate_iou(m1: np.ndarray, m2: np.ndarray):
    """Calculate IoU between two masks."""
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else float(inter / union)


def mask_merge(masks: np.ndarray, confs: np.ndarray, threshold: float = 0.5, not_matched_threshold: float = 1.0):
    """
    Merge overlapping masks based on IoU threshold (HiLo's approach).
    
    Returns:
        pan_mask2: Panoptic mask with merged IDs
        full_remap: Dict mapping all original indices to core indices
        core_remap: Dict mapping core indices to new compact indices
    """
    order = confs.argsort()[::-1]
    cache = {}
    
    def _iou(i, k):
        key = (i, k)
        if key not in cache:
            v = calculate_iou(masks[i], masks[k])
            cache[(i, k)] = cache[(k, i)] = v
        return cache[key]

    pan_mask2 = np.full(masks[0].shape, fill_value=-1, dtype=int)
    remapping = {}
    core_remap = {}
    
    # First pass: identify cores
    for r, i in enumerate(order.tolist()):
        if i in remapping:
            continue
        untouched = pan_mask2 == -1
        new_area = untouched & masks[i]
        if new_area.sum() == 0:
            continue
        pan_mask2[new_area] = len(core_remap)
        core_remap[i] = len(core_remap)

        # Map all later masks that overlap significantly to this core
        for k in order[r+1:].tolist():
            if k not in remapping and _iou(i, k) >= threshold:
                remapping[k] = i

    # Build full remapping
    big_remap = dict(core_remap)
    for orig, better in remapping.items():
        big_remap[orig] = core_remap[better]

    # Optional: assign unmatched to best core if IoU meets threshold
    not_matched = set(range(len(masks))) - set(big_remap.keys())
    for idx in not_matched:
        best_iou, best_core = 0.0, None
        for core in core_remap:
            iou = _iou(idx, core)
            if iou > best_iou:
                best_iou, best_core = iou, core
        if best_iou >= not_matched_threshold and best_core is not None:
            big_remap[idx] = core_remap[best_core]

    return pan_mask2, big_remap, core_remap

# ====================================================================


def load_psg_ground_truth(psg_json_path: str) -> Dict:
    """Load PSG ground truth annotations."""
    print(f"Loading PSG ground truth from {psg_json_path}...")
    with open(psg_json_path, 'r') as f:
        psg_data = json.load(f)
    
    # Create lookup by image_id
    gt_by_image = {}
    for entry in psg_data['data']:
        image_id = entry['image_id']
        gt_by_image[image_id] = entry
    
    return {
        'data': psg_data,
        'by_image': gt_by_image,
        'thing_classes': psg_data['thing_classes'],
        'stuff_classes': psg_data['stuff_classes'],
        'predicate_classes': psg_data['predicate_classes']
    }


def build_hilo_model(cfg_path: str, ckpt_path: str, device: str = 'cuda:0'):
    """Build HiLo model from config and checkpoint."""
    print(f"Loading HiLo model from {ckpt_path}...")
    
    cfg = Config.fromfile(cfg_path)
    
    # Set test mode configuration
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    
    # HiLo: IMPORTANT - Set to use merged output from both decoders
    # This must be set in the bbox_head config BEFORE building the model
    if 'bbox_head' in cfg.model:
        cfg.model.bbox_head.test_forward_output_type = 'merge'
        print(f"  Set test_forward_output_type='merge' in config")
    
    # Set environment variable for evaluation
    os.environ.setdefault("EVAL_PAN_RELS", "True")
    
    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Double-check: set it on the model directly as well
    if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'test_forward_output_type'):
        model.bbox_head.test_forward_output_type = 'merge'
        print(f"  Confirmed: model.bbox_head.test_forward_output_type = '{model.bbox_head.test_forward_output_type}'")
    
    # Load checkpoint
    ckpt = load_checkpoint(model, ckpt_path, map_location='cpu')
    
    # Get classes from checkpoint or use defaults
    if 'meta' in ckpt and 'CLASSES' in ckpt['meta']:
        model.CLASSES = ckpt['meta']['CLASSES']
    elif hasattr(model, 'CLASSES') and model.CLASSES:
        pass
    else:
        # Use default PSG classes
        raise RuntimeError("Model has no CLASSES meta; please provide a class list if needed.")
    
    # Set predicates (HiLo uses OpenPSG predicates)
    if not hasattr(model, 'PREDICATES') or not model.PREDICATES:
        model.PREDICATES = [
            'over','in front of','beside','on','in','attached to','hanging from','on back of','falling off',
            'going down','painted on','walking on','running on','crossing','standing on','lying on',
            'sitting on','flying over','jumping over','jumping from','wearing','holding','carrying',
            'looking at','guiding','kissing','eating','drinking','feeding','biting','catching','picking',
            'playing with','chasing','climbing','cleaning','playing','touching','pushing','pulling',
            'opening','cooking','talking to','throwing','slicing','driving','riding','parked on',
            'driving on','about to hit','kicking','swinging','entering','exiting','enclosing','leaning on',
        ]
    
    model.cfg = cfg
    model.to(device)
    model.eval()
    
    return model


def to_numpy(x):
    """Convert tensor to numpy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        if isinstance(x, list):
            return np.array(x)
        if hasattr(x, 'cpu'):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def convert_psg_to_graph_format(
    psg_entry: Dict,
    thing_classes: List[str],
    stuff_classes: List[str],
    predicate_classes: List[str],
    is_prediction: bool = False,
    image_id: str = None,
) -> Dict:
    """Convert PSG format to graph format."""
    all_classes = thing_classes + stuff_classes
    
    nodes = []
    segment_id_to_node_id = {}
    
    if is_prediction:
        labels = psg_entry.get('labels', [])
        for i, label_id in enumerate(labels):
            node_id = f"obj_{i}"
            label = all_classes[label_id - 1] if label_id > 0 else "unknown"
            nodes.append({"id": node_id, "label": label})
            segment_id_to_node_id[i] = node_id
    else:
        segments_info = psg_entry.get('segments_info', [])
        for i, seg in enumerate(segments_info):
            node_id = f"seg_{seg['id']}"
            cat_id = seg['category_id']
            label = all_classes[cat_id]
            nodes.append({"id": node_id, "label": label})
            segment_id_to_node_id[i] = node_id
    
    edges = []
    
    if is_prediction:
        rel_pairs = psg_entry.get('rel_pair_idxes', [])
        rel_labels = psg_entry.get('rel_labels', [])
        
        for pair_idx, (subj_idx, obj_idx) in enumerate(rel_pairs):
            if pair_idx < len(rel_labels):
                pred_id = rel_labels[pair_idx] - 1  # Convert to 0-based
                
                source_id = segment_id_to_node_id.get(subj_idx, f"obj_{subj_idx}")
                target_id = segment_id_to_node_id.get(obj_idx, f"obj_{obj_idx}")
                relation = predicate_classes[pred_id] if 0 <= pred_id < len(predicate_classes) else f"pred_{pred_id}"
                
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "relation": relation
                })
    else:
        relations = psg_entry.get('relations', [])
        for rel in relations:
            subj_idx, obj_idx, pred_id = rel
            source_id = segment_id_to_node_id.get(subj_idx, f"seg_{subj_idx}")
            target_id = segment_id_to_node_id.get(obj_idx, f"seg_{obj_idx}")
            relation = predicate_classes[pred_id]
            
            edges.append({
                "source": source_id,
                "target": target_id,
                "relation": relation
            })
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def process_images_with_masks(
    model,
    psg_ground_truth: Dict,
    coco_images_dir: str,
    image_range: str = "0-10",
    target_count: int = None,
    masks_output_dir: str = None
):
    """Process images and save masks using HiLo model."""
    
    thing_classes = psg_ground_truth['thing_classes']
    stuff_classes = psg_ground_truth['stuff_classes']
    predicate_classes = psg_ground_truth['predicate_classes']
    
    # Parse image range
    start_idx, end_idx = map(int, image_range.split('-'))
    
    # Get image IDs
    all_image_ids = sorted(psg_ground_truth['by_image'].keys())
    selected_image_ids = [all_image_ids[i] for i in range(start_idx, min(end_idx, len(all_image_ids)))]
    
    predictions_dict = {}
    ground_truth_dict = {}
    masks_dict = {}  # Store prediction masks
    
    print(f"\nProcessing {len(selected_image_ids)} images with HiLo...")
    
    for idx, image_id in enumerate(selected_image_ids):
        print(f"\n[{idx+1}/{len(selected_image_ids)}] Processing image {image_id}...")
        sys.stdout.flush()
        
        if image_id not in psg_ground_truth['by_image']:
            print(f"  Warning: No ground truth found for image {image_id}")
            continue
        
        gt_entry = psg_ground_truth['by_image'][image_id]
        image_filename = gt_entry.get('file_name', 'UNKNOWN')
        
        # Find image path
        candidates = [
            os.path.join(coco_images_dir, image_filename),
            os.path.join(coco_images_dir, os.path.basename(image_filename)),
            os.path.join(coco_images_dir, 'train2017', os.path.basename(image_filename)),
            os.path.join(coco_images_dir, 'val2017', os.path.basename(image_filename)),
        ]
        
        image_path = None
        for c in candidates:
            if os.path.exists(c):
                image_path = c
                break
        
        if image_path is None:
            print(f"  Warning: Image not found")
            continue
        
        print(f"  Found image: {image_path}")
        
        # Run inference with HiLo
        try:
            t0 = time.perf_counter()
            result = inference_detector(model, image_path)
            
            # Handle result format (can be list or single result)
            if isinstance(result, (list, tuple)):
                result = result[0]
            
            infer_dt = time.perf_counter() - t0
            
            # Extract data from HiLo Result object
            labels = to_numpy(result.labels) if hasattr(result, 'labels') else np.array([])
            rel_pairs = to_numpy(result.rel_pair_idxes) if hasattr(result, 'rel_pair_idxes') else np.array([])
            rel_labels = to_numpy(result.rel_labels) if hasattr(result, 'rel_labels') else np.array([])
            
            # For HiLo, rel_scores might be in triplet_scores or rel_dists
            if hasattr(result, 'rel_scores') and result.rel_scores is not None:
                rel_scores = to_numpy(result.rel_scores)
            elif hasattr(result, 'triplet_scores') and result.triplet_scores is not None:
                rel_scores = to_numpy(result.triplet_scores)
            else:
                rel_scores = np.ones(len(rel_labels), dtype=np.float32)
            
            # Extract masks
            masks = to_numpy(result.masks) if hasattr(result, 'masks') and result.masks is not None else None
            
            # Extract bounding boxes if available
            bboxes = to_numpy(result.refine_bboxes) if hasattr(result, 'refine_bboxes') and result.refine_bboxes is not None else None
            
            if masks is not None:
                print(f"  Extracted masks: shape={masks.shape} (before merge)")
            else:
                print(f"  Warning: No masks in result")
            
            print(f"  Inference: {infer_dt:.3f}s | {len(labels)} objects (before merge) | {len(rel_pairs)} relations")
            
            # ===== APPLY MASK MERGE (HiLo's approach) =====
            if masks is not None and len(masks) > 0 and bboxes is not None:
                # Get confidence scores from bboxes
                confs = bboxes[:, -1] if bboxes.shape[1] >= 5 else np.ones((len(labels),), np.float32)
                
                # Apply mask merge with IoU threshold
                print(f"  Applying mask_merge (IoU threshold=0.5)...")
                pan_mask2, full_remap, core_remap = mask_merge(
                    masks.astype(bool), confs, threshold=0.5, not_matched_threshold=1.0
                )
                
                # Remap labels and bboxes to only keep cores
                K = len(core_remap)
                new_labels = np.empty((K,), dtype=labels.dtype)
                new_bboxes = np.empty((K, bboxes.shape[1]), dtype=bboxes.dtype)
                
                for orig, new_id in core_remap.items():
                    new_labels[new_id] = labels[orig]
                    new_bboxes[new_id] = bboxes[orig]
                
                # Remap relations through full_remap
                new_pairs = []
                new_rel_labels_list = []
                for (s, o), rl in zip(rel_pairs, rel_labels):
                    s, o = int(s), int(o)
                    if s in full_remap and o in full_remap:
                        new_pairs.append((full_remap[s], full_remap[o]))
                        new_rel_labels_list.append(rl)
                
                labels = new_labels
                bboxes = new_bboxes
                rel_pairs = np.array(new_pairs, dtype=np.int32) if new_pairs else np.zeros((0, 2), dtype=np.int32)
                rel_labels = np.array(new_rel_labels_list, dtype=rel_labels.dtype) if new_rel_labels_list else np.array([], dtype=rel_labels.dtype)
                
                print(f"  After merge: {len(labels)} nodes, {len(rel_pairs)} relations")
            
            # Convert to graph format
            pred_entry = {
                'labels': labels,
                'rel_pair_idxes': rel_pairs,
                'rel_labels': rel_labels,
            }
            
            pred_graph = convert_psg_to_graph_format(
                pred_entry, thing_classes, stuff_classes, predicate_classes,
                is_prediction=True, image_id=image_id
            )
            
            gt_graph = convert_psg_to_graph_format(
                gt_entry, thing_classes, stuff_classes, predicate_classes,
                is_prediction=False, image_id=image_id
            )
            
            predictions_dict[image_id] = pred_graph
            ground_truth_dict[image_id] = gt_graph
            
            # Save masks
            if masks is not None and masks_output_dir is not None:
                masks_dict[image_id] = {
                    'masks': masks,
                    'labels': labels,
                    'node_ids': [node['id'] for node in pred_graph['nodes']],
                    'bboxes': bboxes,
                }
            
            print(f"  Prediction: {len(pred_graph['nodes'])} nodes, {len(pred_graph['edges'])} edges")
            print(f"  Ground Truth: {len(gt_graph['nodes'])} nodes, {len(gt_graph['edges'])} edges")
            
            if target_count and len(predictions_dict) >= target_count:
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            continue
    
    return predictions_dict, ground_truth_dict, masks_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene graphs WITH masks from HiLo"
    )
    parser.add_argument('--config', type=str, required=True, help='HiLo config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--psg-json', type=str, required=True, help='PSG annotations JSON')
    parser.add_argument('--coco-images', type=str, required=True, help='COCO images directory')
    parser.add_argument('--output-dir', type=str, default='./hilo_graphs', help='Output directory')
    parser.add_argument('--image-range', type=str, default='0-10', help='Image index range (e.g., "0-10")')
    parser.add_argument('--target-count', type=int, default=None, help='Stop after N successful images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Load model
    model = build_hilo_model(args.config, args.checkpoint, device=args.device)
    
    # Load PSG
    psg_data = load_psg_ground_truth(args.psg_json)
    
    # Process images
    predictions, ground_truth, masks = process_images_with_masks(
        model, psg_data, args.coco_images,
        image_range=args.image_range,
        target_count=args.target_count,
        masks_output_dir=masks_dir
    )
    
    # Save scene graphs
    output_data = {
        'metadata': {
            'model': 'HiLo',
            'config': args.config,
            'checkpoint': args.checkpoint,
            'psg_json': args.psg_json,
            'image_range': args.image_range,
            'thing_classes': psg_data['thing_classes'],
            'stuff_classes': psg_data['stuff_classes'],
            'predicate_classes': psg_data['predicate_classes'],
        },
        'images': {
            img_id: {
                'prediction': predictions[img_id],
                'ground_truth': ground_truth[img_id]
            }
            for img_id in predictions.keys()
        }
    }
    
    range_str = args.image_range.replace('-', '_')
    output_path = os.path.join(args.output_dir, f'scene_graphs_{range_str}.json')
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved scene graphs to: {output_path}")
    
    # Save masks separately (as pickle because they're large numpy arrays)
    if masks:
        masks_path = os.path.join(masks_dir, f'prediction_masks_{range_str}.pkl')
        with open(masks_path, 'wb') as f:
            pickle.dump(masks, f)
        print(f"Saved prediction masks to: {masks_path}")
    
    print(f"\nProcessed {len(predictions)} images successfully!")


if __name__ == '__main__':
    main()
