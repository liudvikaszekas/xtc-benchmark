"""
Enhanced script to generate scene graphs WITH prediction masks saved.

This extends generate_openpsg_graphs.py to also save the prediction masks
so they can be visualized later alongside ground truth masks.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle

# Add OpenPSG to path
sys.path.insert(0, '/home/mpws25/OpenPSG')

import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
import time
import traceback


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
        rel_dists = psg_entry.get('rel_dists', [])
        
        for pair_idx, (subj_idx, obj_idx) in enumerate(rel_pairs):
            if pair_idx < len(rel_dists):
                pred_scores = rel_dists[pair_idx, 1:]
                pred_id = np.argmax(pred_scores)
                
                source_id = segment_id_to_node_id.get(subj_idx, f"obj_{subj_idx}")
                target_id = segment_id_to_node_id.get(obj_idx, f"obj_{obj_idx}")
                relation = predicate_classes[pred_id]
                
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
    """Process images and save masks."""
    
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
    
    print(f"\nProcessing {len(selected_image_ids)} images...")
    
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
        
        # Run inference
        try:
            t0 = time.perf_counter()
            result = inference_detector(model, image_path)
            infer_dt = time.perf_counter() - t0
            
            # Extract data
            labels = result.labels
            rel_pairs = result.rel_pair_idxes
            rel_dists = result.rel_dists
            masks = result.masks  # This is the key addition!
            
            # Convert to numpy
            labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
            rel_pairs_np = rel_pairs.cpu().numpy() if hasattr(rel_pairs, 'cpu') else np.array(rel_pairs)
            rel_dists_np = rel_dists.cpu().numpy() if hasattr(rel_dists, 'cpu') else np.array(rel_dists)
            
            # masks is already numpy
            if masks is not None:
                masks_np = masks
                print(f"  Extracted masks: shape={masks_np.shape}")
            else:
                masks_np = None
                print(f"  Warning: No masks in result")
            
            print(f"  Inference: {infer_dt:.3f}s | {len(labels_np)} objects | {len(rel_pairs_np)} relations")
            
            # Convert to graph format
            pred_entry = {
                'labels': labels_np,
                'rel_pair_idxes': rel_pairs_np,
                'rel_dists': rel_dists_np,
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
            if masks_np is not None and masks_output_dir is not None:
                masks_dict[image_id] = {
                    'masks': masks_np,
                    'labels': labels_np,
                    'node_ids': [node['id'] for node in pred_graph['nodes']]
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
        description="Generate scene graphs WITH masks from OpenPSG"
    )
    parser.add_argument('--config', type=str, required=True, help='OpenPSG config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--psg-json', type=str, required=True, help='PSG annotations JSON')
    parser.add_argument('--coco-images', type=str, required=True, help='COCO images directory')
    parser.add_argument('--output-dir', type=str, default='./openpsg_graphs', help='Output directory')
    parser.add_argument('--image-range', type=str, default='0-10', help='Image index range (e.g., "0-10")')
    parser.add_argument('--target-count', type=int, default=None, help='Stop after N successful images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    
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
