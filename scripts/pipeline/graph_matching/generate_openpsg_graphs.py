"""
Script to generate scene graphs using OpenPSG from COCO images and compare with ground truth.

This script:
1. Loads OpenPSG model
2. Loads PSG ground truth annotations
3. Runs inference on specified COCO images
4. Saves both predicted and ground truth scene graphs in JSON format
5. Format is compatible with hungarian.py matching algorithm
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

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
    image_id: Optional[str] = None,
) -> Dict:
    """
    Convert PSG format to graph format compatible with hungarian.py
    
    PSG format has:
    - segments_info: list of segments with id, category_id, etc.
    - relations: list of [subject_idx, object_idx, predicate_id]
    
    Graph format needs:
    - nodes: [{id, label}]
    - edges: [{source, target, relation}]
    """
    all_classes = thing_classes + stuff_classes
    
    # Build nodes from segments_info
    nodes = []
    segment_id_to_node_id = {}
    
    if is_prediction:
        # For predictions, we have labels array
        labels = psg_entry.get('labels', [])
        for i, label_id in enumerate(labels):
            node_id = f"obj_{i}"
            # label_id is 1-indexed in PSG, 0-indexed in class list
            label = all_classes[label_id - 1] if label_id > 0 else "unknown"
            nodes.append({"id": node_id, "label": label})
            segment_id_to_node_id[i] = node_id
    else:
        # For ground truth, we have segments_info
        segments_info = psg_entry.get('segments_info', [])
        for i, seg in enumerate(segments_info):
            node_id = f"seg_{seg['id']}"
            cat_id = seg['category_id']
            label = all_classes[cat_id]
            nodes.append({"id": node_id, "label": label})
            segment_id_to_node_id[i] = node_id
    
    # Build edges from relations
    edges = []
    relations = psg_entry.get('relations', [])
    
    if is_prediction:
        # For predictions: rel_pair_idxes and rel_dists
        rel_pairs = psg_entry.get('rel_pair_idxes', [])
        rel_dists = psg_entry.get('rel_dists', [])
        
        for pair_idx, (subj_idx, obj_idx) in enumerate(rel_pairs):
            if pair_idx < len(rel_dists):
                # Get the predicate with highest score (skip background at index 0)
                pred_scores = rel_dists[pair_idx, 1:]  # Skip background
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
        # For ground truth: relations is [subject_idx, object_idx, predicate_id]
        for rel in relations:
            subj_idx, obj_idx, pred_id = rel
            source_id = segment_id_to_node_id.get(subj_idx, f"seg_{subj_idx}")
            target_id = segment_id_to_node_id.get(obj_idx, f"seg_{obj_idx}")
            # pred_id is 1-indexed in PSG
            relation = predicate_classes[pred_id - 1] if pred_id > 0 else "unknown"
            
            edges.append({
                "source": source_id,
                "target": target_id,
                "relation": relation
            })

    # Debug printing: show counts (include image id if available)
    _img = f" image_id={image_id}" if image_id is not None else ""
    kind = "PREDICTION" if is_prediction else "GROUND-TRUTH"
    try:
        print(f"    [convert] {kind}{_img}: built {len(nodes)} nodes and {len(edges)} edges")
    except Exception:
        pass

    return {
        "nodes": nodes,
        "edges": edges
    }


def run_inference_and_collect(
    model,
    config,
    coco_images_dir: str,
    psg_ground_truth: Dict,
    image_ids: List[str],
    output_dir: str,
    target_count: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """
    Run inference on specified images and collect both predictions and ground truth.
    
    Returns:
        predictions_dict: {image_id: graph_format}
        ground_truth_dict: {image_id: graph_format}
    """
    predictions_dict = {}
    ground_truth_dict = {}
    
    thing_classes = psg_ground_truth['thing_classes']
    stuff_classes = psg_ground_truth['stuff_classes']
    predicate_classes = psg_ground_truth['predicate_classes']
    
    if target_count is None:
        print(f"\nProcessing {len(image_ids)} images...")
    else:
        print(f"\nProcessing starting from list of {len(image_ids)} candidate images, aiming to collect {target_count} existing images...")

    for idx, image_id in enumerate(image_ids):
        iter_start = time.perf_counter()
        print(f"[{idx+1}/{len(image_ids)}] Processing image {image_id}... (iter_start={iter_start:.3f})", flush=True)

        # Get ground truth entry
        if image_id not in psg_ground_truth['by_image']:
            print(f"  Warning: No ground truth found for image {image_id}", flush=True)
            continue

        gt_entry = psg_ground_truth['by_image'][image_id]

        # Get image path (try multiple common location patterns)
        image_filename = gt_entry.get('file_name', 'UNKNOWN')
        print(f"  file_name: {image_filename}", flush=True)
        lookup_start = time.perf_counter()

        candidates = []
        # If file_name already contains subdir (e.g., 'val2017/000000123.jpg'), join directly
        candidates.append(os.path.join(coco_images_dir, image_filename))
        # Try basename in coco root
        candidates.append(os.path.join(coco_images_dir, os.path.basename(image_filename)))
        # Try common subfolders
        candidates.append(os.path.join(coco_images_dir, 'train2017', os.path.basename(image_filename)))
        candidates.append(os.path.join(coco_images_dir, 'val2017', os.path.basename(image_filename)))

        image_path = None
        for c in candidates:
            if os.path.exists(c):
                image_path = c
                break

        lookup_dt = time.perf_counter() - lookup_start
        if image_path is None:
            print(f"  Warning: Image not found. Tried: {candidates} | lookup_time={lookup_dt:.4f}s", flush=True)
            # skip but keep scanning until we hit target_count if requested
            continue
        else:
            print(f"  Found image at: {image_path} | lookup_time={lookup_dt:.4f}s", flush=True)

        # Run inference
        try:
            t0 = time.perf_counter()
            print(f"  Starting inference for {image_id}...", flush=True)
            result = inference_detector(model, image_path)
            infer_dt = time.perf_counter() - t0
            print(f"  Inference finished (dt={infer_dt:.3f}s)", flush=True)

            # Extract prediction arrays safely and log shapes
            try:
                labels = result.labels
            except Exception:
                labels = []
            try:
                rel_pairs = result.rel_pair_idxes
            except Exception:
                rel_pairs = []
            try:
                rel_dists = result.rel_dists
            except Exception:
                rel_dists = []

            # Convert tensors to numpy if needed and get shapes
            try:
                labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
            except Exception:
                labels_np = np.array(labels)
            try:
                rel_pairs_np = rel_pairs.cpu().numpy() if hasattr(rel_pairs, 'cpu') else np.array(rel_pairs)
            except Exception:
                rel_pairs_np = np.array(rel_pairs)
            try:
                rel_dists_np = rel_dists.cpu().numpy() if hasattr(rel_dists, 'cpu') else np.array(rel_dists)
            except Exception:
                rel_dists_np = np.array(rel_dists)

            print(f"  Inference time: {infer_dt:.3f}s | labels: {getattr(labels_np,'shape', getattr(labels_np,'size', 'N/A'))} | rel_pairs: {getattr(rel_pairs_np,'shape', getattr(rel_pairs_np,'size', 'N/A'))} | rel_dists: {getattr(rel_dists_np,'shape', getattr(rel_dists_np,'size', 'N/A'))}", flush=True)

            # Convert prediction to graph format (pass image id for debugging)
            pred_entry = {
                'labels': labels_np,
                'rel_pair_idxes': rel_pairs_np,
                'rel_dists': rel_dists_np,
            }

            conv_start = time.perf_counter()
            pred_graph = convert_psg_to_graph_format(
                pred_entry, thing_classes, stuff_classes, predicate_classes, is_prediction=True, image_id=image_id
            )

            # Convert ground truth to graph format (pass image id for debugging)
            gt_graph = convert_psg_to_graph_format(
                gt_entry, thing_classes, stuff_classes, predicate_classes, is_prediction=False, image_id=image_id
            )
            conv_dt = time.perf_counter() - conv_start
            print(f"  Conversion finished (dt={conv_dt:.4f}s)", flush=True)

            predictions_dict[image_id] = pred_graph
            ground_truth_dict[image_id] = gt_graph

            # Report collected count
            print(f"  Collected {len(predictions_dict)} images so far (target={target_count})", flush=True)

            # If user asked to collect a target_count of existing images, stop when reached
            if target_count is not None and len(predictions_dict) >= target_count:
                print(f"Collected target of {target_count} images; stopping scan.", flush=True)
                break

            # Extra detail: show predicate distribution in top-k (if available)
            try:
                if rel_dists_np is not None and getattr(rel_dists_np, 'ndim', 0) == 2:
                    # skip background (index 0) when reporting
                    rel_scores = rel_dists_np[:, 1:]
                    top_k = min(5, rel_scores.shape[0])
                    if top_k > 0:
                        top_idx = np.argsort(rel_scores.max(1))[-top_k:][::-1]
                        top_preds = [predicate_classes[np.argmax(rel_scores[i])] for i in top_idx]
                        print(f"  Top {top_k} predicted predicates: {top_preds}", flush=True)
            except Exception:
                pass

            print(f"  Prediction: {len(pred_graph['nodes'])} nodes, {len(pred_graph['edges'])} edges", flush=True)
            print(f"  Ground Truth: {len(gt_graph['nodes'])} nodes, {len(gt_graph['edges'])} edges", flush=True)

        except Exception as e:
            print(f"  Error processing image {image_id}: {e}", flush=True)
            traceback.print_exc()
            continue
    
    return predictions_dict, ground_truth_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene graphs using OpenPSG and compare with ground truth"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/home/mpws25/OpenPSG/configs/psgtr/psgtr_r50_psg.py',
        help='Path to OpenPSG config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/mpws25/OpenPSG/RESULTS/epoch_60.pth',
        help='Path to OpenPSG checkpoint file'
    )
    parser.add_argument(
        '--psg-json',
        type=str,
        default='/home/mpws25/OneDrive_1_11-4-2025/psg/psg.json',
        help='Path to PSG ground truth JSON file'
    )
    parser.add_argument(
        '--coco-images',
        type=str,
        default='/home/mpws25/datasets/coco/coco',
        help='Path to COCO images directory (should contain train2017/ and val2017/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/mpws25/graph_matching/openpsg_graphs',
        help='Output directory for generated graphs'
    )
    parser.add_argument(
        '--image-range',
        type=str,
        default='0-50',
        help='Range of images to process (e.g., "0-100" or "100-200")'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse image range
    start_idx, end_idx = map(int, args.image_range.split('-'))
    
    # Load PSG ground truth
    psg_gt = load_psg_ground_truth(args.psg_json)
    
    # Get image IDs and interpret the provided range as: start_index - target_count
    all_image_ids = sorted(psg_gt['by_image'].keys())
    start_idx, target_count = map(int, args.image_range.split('-'))

    # Candidate list is the remaining PSG entries starting from start_idx
    image_ids = all_image_ids[start_idx:]

    print(f"Scanning PSG entries starting at index {start_idx}, aiming to collect {target_count} existing images (candidates: {len(image_ids)})")
    print(f"Sample candidate image IDs: {image_ids[:5]}")
    
    # Load OpenPSG model
    print(f"\nLoading OpenPSG model from {args.checkpoint}...")
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    
    # Run inference and collect results
    # Run inference and collect results until we have target_count valid images (or exhaust candidates)
    predictions, ground_truths = run_inference_and_collect(
        model=model,
        config=cfg,
        coco_images_dir=args.coco_images,
        psg_ground_truth=psg_gt,
        image_ids=image_ids,
        output_dir=args.output_dir,
        target_count=target_count
    )
    
    # Save results
    output_file = os.path.join(
        args.output_dir, 
        f'scene_graphs_{start_idx}_{end_idx}.json'
    )
    
    output_data = {
        'metadata': {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'psg_json': args.psg_json,
            'image_range': args.image_range,
            'num_images': len(image_ids),
            'target_images_requested': target_count,
            'thing_classes': psg_gt['thing_classes'],
            'stuff_classes': psg_gt['stuff_classes'],
            'predicate_classes': psg_gt['predicate_classes']
        },
        'images': {}
    }
    
    for image_id in predictions.keys():
        output_data['images'][image_id] = {
            'prediction': predictions[image_id],
            'ground_truth': ground_truths[image_id]
        }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Scene graph generation complete!")
    print(f"{'='*70}")
    print(f"Total images processed: {len(predictions)}")
    print(f"Output saved to: {output_file}")
    print(f"\nFormat details:")
    print(f"  - Each entry has 'prediction' and 'ground_truth' scene graphs")
    print(f"  - Nodes have: id, label")
    print(f"  - Edges have: source, target, relation")
    print(f"  - Compatible with hungarian.py")
    
    # Print sample
    if predictions:
        sample_id = list(predictions.keys())[0]
        print(f"\nSample graph for image {sample_id}:")
        print(f"  Prediction nodes: {len(predictions[sample_id]['nodes'])}")
        print(f"  Prediction edges: {len(predictions[sample_id]['edges'])}")
        print(f"  Ground truth nodes: {len(ground_truths[sample_id]['nodes'])}")
        print(f"  Ground truth edges: {len(ground_truths[sample_id]['edges'])}")


if __name__ == '__main__':
    main()
