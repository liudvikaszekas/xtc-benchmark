#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

def run_command(cmd: List[str], description: str) -> bool:
    print(f"Running {description}...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {description}: {e}")
        return False

def convert_merged_to_scene_graph(merged_graph_path: Path, output_dir: Path, label: str) -> bool:
    """
    Convert merged graph format to scene graph format.
    Input format: {image_id: {groups: [...], edges: [...]}}
    Output format: {image_id, file_name, boxes: [{index, label, bbox_xyxy}], relations: [...]}
    """
    with open(merged_graph_path, 'r') as f:
        merged_data = json.load(f)
    
    # JSONL Setup
    jsonl_path = output_dir / "merged_graphs.jsonl"
    try:
        from utils import append_jsonl
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import append_jsonl

    for image_id_str, graph_data in merged_data.items():
        image_id = graph_data.get('image_id')
        file_name = graph_data.get('file_name')
        groups = graph_data.get('groups', [])
        edges = graph_data.get('edges', [])
        
        group_id_to_index = {}
        boxes = []
        
        for idx, group in enumerate(sorted(groups, key=lambda g: g.get('group_id', 0))):
            group_id = group.get('group_id')
            group_label = group.get('label', '')
            bbox = group.get('bbox', [])
            
            if len(bbox) == 4:
                bbox_xyxy = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            else:
                bbox_xyxy = [0, 0, 0, 0]
            
            group_id_to_index[group_id] = idx
            seg_ids = group.get('seg_ids', [])
            boxes.append({
                "index": idx,
                "category_id": group.get('category_id'),
                "label": group_label,
                "bbox_xyxy": bbox_xyxy,
                "seg_ids": seg_ids,
                "id": seg_ids[0] if seg_ids else None
            })
        
        relations = []
        for edge in edges:
            sub_gid = edge.get('subject_group_id')
            obj_gid = edge.get('object_group_id')
            
            if sub_gid not in group_id_to_index or obj_gid not in group_id_to_index:
                continue
            
            sub_idx = group_id_to_index[sub_gid]
            obj_idx = group_id_to_index[obj_gid]
            
            sub_label = boxes[sub_idx].get('label', '')
            obj_label = boxes[obj_idx].get('label', '')
            sub_seg_ids = boxes[sub_idx].get('seg_ids', [])
            obj_seg_ids = boxes[obj_idx].get('seg_ids', [])

            sub_spec_id = edge.get('subject_seg_id')
            obj_spec_id = edge.get('object_seg_id')
            
            relation = {
                "subject_index": sub_idx,
                "subject_label": sub_label,
                "subject_seg_ids": sub_seg_ids,
                "subject_id": sub_spec_id if sub_spec_id is not None else (sub_seg_ids[0] if sub_seg_ids else None),
                "object_index": obj_idx,
                "object_label": obj_label,
                "object_seg_ids": obj_seg_ids,
                "object_id": obj_spec_id if obj_spec_id is not None else (obj_seg_ids[0] if obj_seg_ids else None),
                "predicate": edge.get('best_predicate', ''),
                "predicate_score": edge.get('best_predicate_score', 0.0),
                "no_relation_score": edge.get('no_relation_score', 0.0)
            }
            
            relations.append(relation)
        
        scene_graph = {
            "image_id": image_id,
            "file_name": file_name,
            "boxes": boxes,
            "relations": relations
        }
        
        output_file = output_dir / f"scene-graph_{image_id}.json"
        with open(output_file, 'w') as f:
            json.dump(scene_graph, f, indent=2)
        
        # Append to JSONL
        append_jsonl(jsonl_path, scene_graph)
        
    print(f"Converted {len(merged_data)} merged entries.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Graph Merging Pipeline")
    parser.add_argument("--anno-json", required=True, help="Path to segmentation anno.json")
    parser.add_argument("--scene-graph-pkl", required=True, help="Path to scene-graph.pkl")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--clean-relations-dir", help="Directory containing cleaned scene-graph IDs")
    parser.add_argument("--agg", default="mean")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--predicate-threshold", type=float, default=0.8)
    parser.add_argument("--min-group-size", type=int, default=3, help="Minimum group size for merging")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anno_path = Path(args.anno_json)
    
    # 1. merge_graph.py
    # Output: anno_merged.json (initially next to anno.json with _merged suffix)
    # The script outputs to same dir as input, but with suffix.
    # We should move it or rely on it.
    
    cmd1 = [
        "python", str(script_dir / "merge_graph.py"),
        "--input", str(anno_path),
        "--padding", str(args.padding),
        "--min-group-size", str(args.min_group_size)
    ]
    if not run_command(cmd1, "merge_graph.py"): return
    
    # Expected output of step 1
    temp_merged_path = anno_path.parent / anno_path.name.replace('.json', '_merged.json')
    anno_merged_path = out_dir / 'anno_merged.json'
    
    if temp_merged_path.exists():
        shutil.move(str(temp_merged_path), str(anno_merged_path))
    else:
        print(f"Error: Expected merged file not found at {temp_merged_path} or {anno_merged_path}")
        # Note: if output dir is same as input dir, it might be already there.
        if anno_merged_path.exists():
            pass # OK
        else:
            return

    # 2. merge_masks.py
    merged_masks_dir = out_dir / 'merged_segmentations'
    cmd2 = [
        "python", str(script_dir / "merge_masks.py"),
        "--anno", str(anno_path),
        "--merged", str(anno_merged_path),
        "--output-dir", str(merged_masks_dir)
    ]
    if not run_command(cmd2, "merge_masks.py"): return

    # 3. merge_edges.py
    merged_graph_path = out_dir / 'anno_merged_edges.json'
    cmd3 = [
        "python", str(script_dir / "merge_edges.py"),
        "--anno", str(anno_path),
        "--merged", str(anno_merged_path),
        "--scene-pkl", args.scene_graph_pkl,
        "--output", str(merged_graph_path),
        "--format", "json",
        "--agg", args.agg,
        "--threshold", str(args.threshold),
        "--predicate-threshold", str(args.predicate_threshold)
    ]
    if args.clean_relations_dir:
         cmd3.extend(["--relations-json-dir", args.clean_relations_dir])
         
    if not run_command(cmd3, "merge_edges.py"): return

    # 4. Convert
    convert_merged_to_scene_graph(merged_graph_path, out_dir, "GT")
    
    print("Graph Merging Complete.")

if __name__ == "__main__":
    main()
