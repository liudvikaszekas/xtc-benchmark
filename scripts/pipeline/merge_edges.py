#!/usr/bin/env python3
"""
Build anno_merged_edges.{json|pkl}

Inputs:
- anno.json          (from segmentation pipeline)
- anno_merged.json   (old_seg_id -> merged group_id per image)
- scene-graph.pkl    (output of fair_psgg inference)

Output:
- anno_merged_edges.{json|pkl} with:
  - groups: merged instance nodes (one per group_id)
  - edges: relations between those groups, with full predicate score vectors

Optional flags:
  --format json|pkl   (output format, default: json)
  --relations-json-dir DIR (optional, if provided, load relations from these JSONs instead of PKL scores)
  --format json|pkl   (output format, default: json)
  --agg mean|max      (aggregation method for merging edge scores, default: mean)
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np


def merge_boxes(bboxes):
    xs1 = [b[0] for b in bboxes]
    ys1 = [b[1] for b in bboxes]
    xs2 = [b[2] for b in bboxes]
    ys2 = [b[3] for b in bboxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_class_maps(anno_data):
    """
    Build mapping:
      (type, category_id) -> label
    and inverse:
      label -> [category_id]   (global category ids, thing+stuff)
    """
    thing_classes = anno_data["thing_classes"]
    stuff_classes = anno_data["stuff_classes"]

    label_names = thing_classes + stuff_classes
    n_thing = len(thing_classes)

    class_map = {}
    label_to_cids = defaultdict(list)

    for cid, name in enumerate(label_names):
        t = "thing" if cid < n_thing else "stuff"
        class_map[(t, cid)] = name
        label_to_cids[name].append(cid)

    return class_map, label_to_cids


def compute_groups_for_image(image_data, id_mapping):
    """
    From anno.json + id_mapping, build merged groups:
      group_id -> { 'group_id', 'category_id', 'isthing', 'bbox', 'area', 'score' }
    """
    segs = image_data["segments_info"]
    anns = image_data["annotations"]

    grouped = defaultdict(list)  # group_id -> list of (seg, ann)
    for seg, ann in zip(segs, anns):
        old_id = seg["id"]
        new_id = id_mapping.get(str(old_id), id_mapping.get(old_id, old_id))
        new_id = int(new_id)
        grouped[new_id].append((seg, ann))

    groups = {}
    for gid, items in grouped.items():
        segs_g = [s for s, _ in items]
        anns_g = [a for _, a in items]

        base_seg = segs_g[0]
        cat_id = base_seg["category_id"]
        isthing = int(base_seg["isthing"])

        # sanity: all same category
        assert all(s["category_id"] == cat_id for s in segs_g)

        area = sum(s.get("area", 0) for s in segs_g)
        score = max(s.get("score", 0.0) for s in segs_g)

        bboxes = [a["bbox"] for a in anns_g]
        merged_bbox = merge_boxes(bboxes)

        groups[gid] = {
            "group_id": gid,
            "category_id": cat_id,
            "isthing": isthing,
            "bbox": merged_bbox,
            "area": area,
            "score": score,
            "seg_ids": [s["id"] for s in segs_g],
            "segments_detail": [
                {"id": s["id"], "bbox": a["bbox"]} 
                for s, a in zip(segs_g, anns_g)
            ],
            # 'label' filled later
        }

    return groups


def assign_boxes_to_groups(boxes, groups, label_to_cids, iou_thresh=0.1):
    """
    For each scene-graph box, find best-matching group_id (by label + IoU).
    boxes: list of dicts with fields: index, label, bbox_xyxy
    Returns: dict box_index -> group_id or None.
    """
    box_to_group = {}

    for b in boxes:
        idx = b["index"]
        label = b["label"]
        box = list(map(float, b["bbox_xyxy"]))

        cand_cids = label_to_cids.get(label, [])

        best_gid = None
        best_iou = 0.0
        for gid, g in groups.items():
            if g["category_id"] not in cand_cids:
                continue
            iou = iou_xyxy(box, g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gid = gid

        if best_gid is not None and best_iou >= iou_thresh:
            box_to_group[idx] = best_gid
        else:
            box_to_group[idx] = None

    return box_to_group


def build_merged_edges_from_pkl(
    pairs,
    rel_scores,
    box_to_group,
    predicate_classes,
    skip_self=True,
    agg="mean",  # "mean" or "max"
    threshold=None,
    predicate_threshold=None,
):
    """
    pairs: (num_pairs, 2) array of (sbj_idx, obj_idx)
    rel_scores: (num_pairs, num_predicates+1) scores (0 = no_relation)
    box_to_group: dict box_idx -> group_id (or None)
    predicate_classes: list of predicate names, length = num_predicates
    threshold: if set, skip pairs with no_relation_score > threshold
    predicate_threshold: if set, skip pairs with max(predicate_scores) < predicate_threshold
    """
    merged = {}  # (sub_gid, obj_gid) -> {"sum": np.array, "count": int} or {"max": np.array}

    num_pairs = len(pairs)
    for i in range(num_pairs):
        sbj_idx, obj_idx = map(int, pairs[i])
        sbj_gid = box_to_group.get(sbj_idx)
        obj_gid = box_to_group.get(obj_idx)
        if sbj_gid is None or obj_gid is None:
            continue
        if skip_self and sbj_gid == obj_gid:
            continue

        scores = np.asarray(rel_scores[i], dtype=float)  # shape: (num_predicates+1,)
        
        if threshold is not None and scores[0] > threshold:
            continue

        if predicate_threshold is not None:
             # indices 1..end are the actual relations
            if np.max(scores[1:]) < predicate_threshold:
                continue

        # Score used to determine 'best' pair representative
        curr_quality = float(np.max(scores[1:]))

        key = (sbj_gid, obj_gid)

        if agg == "mean":
            if key not in merged:
                merged[key] = {
                    "sum": scores.copy(), 
                    "count": 1,
                    "best_pair_indices": (sbj_idx, obj_idx),
                    "best_pair_score": curr_quality
                }
            else:
                merged[key]["sum"] += scores
                merged[key]["count"] += 1
                if curr_quality > merged[key]["best_pair_score"]:
                    merged[key]["best_pair_indices"] = (sbj_idx, obj_idx)
                    merged[key]["best_pair_score"] = curr_quality
        elif agg == "max":
            if key not in merged:
                merged[key] = {
                    "max": scores.copy(),
                    "best_pair_indices": (sbj_idx, obj_idx),
                    "best_pair_score": curr_quality
                }
            else:
                merged[key]["max"] = np.maximum(merged[key]["max"], scores)
                if curr_quality > merged[key]["best_pair_score"]:
                     merged[key]["best_pair_indices"] = (sbj_idx, obj_idx)
                     merged[key]["best_pair_score"] = curr_quality
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    merged_edges = []
    for (sub_gid, obj_gid), v in merged.items():
        if agg == "mean":
            avg = v["sum"] / float(v["count"])
        else:
            avg = v["max"]

        avg = avg.tolist()

        no_rel = float(avg[0])
        pred_scores = avg[1:]  # aligned with predicate_classes

        # map predicate name -> score
        pred_score_dict = {
            predicate_classes[i]: float(pred_scores[i])
            for i in range(len(predicate_classes))
        }

        # best predicate (ignoring "no relation")
        best_idx = int(np.argmax(pred_scores))
        best_pred = predicate_classes[best_idx]
        best_score = float(pred_scores[best_idx])
        
        sbj_box_idx, obj_box_idx = v["best_pair_indices"]

        merged_edges.append({
            "subject_group_id": sub_gid,
            "object_group_id": obj_gid,
            "no_relation_score": no_rel,
            "predicate_scores": pred_score_dict,
            "best_predicate": best_pred,
            "best_predicate_score": best_score,
            "subject_box_index": sbj_box_idx,
            "object_box_index": obj_box_idx
        })

    return merged_edges


def build_merged_edges_from_json(
    relations_list: List[Dict[str, Any]],
    box_to_group: Dict[int, int],
    skip_self=True
):
    """
    Build merged edges from clean/refined sparse JSON format.
    
    relations_list: list of relation dicts with 'predicates' list
    box_to_group: dict box_idx -> group_id
    """
    merged = defaultdict(lambda: {
        "predicates": defaultdict(float), # predicate -> max_score
        "no_relation_scores": [],
        "best_pair_indices": None,
        "best_pair_score": -1.0
    })

    for rel in relations_list:
        sbj_idx = rel['subject_index']
        obj_idx = rel['object_index']
        
        sbj_gid = box_to_group.get(sbj_idx)
        obj_gid = box_to_group.get(obj_idx)
        
        if sbj_gid is None or obj_gid is None:
            continue
        if skip_self and sbj_gid == obj_gid:
            continue

        key = (sbj_gid, obj_gid)
        
        # Track no_relation score (average it later?)
        no_rel = rel.get('no_relation_score', 0.0)
        merged[key]["no_relation_scores"].append(no_rel)
        
        # Process predicates
        # We take the union of predicates. If duplicate, take max score.
        current_max_score = 0.0
        
        for p in rel.get('predicates', []):
            pred_name = p['predicate']
            score = p['predicate_score']
            # validated = p.get('validated', True) # Assume only validated are in the list
            
            merged[key]["predicates"][pred_name] = max(merged[key]["predicates"][pred_name], score)
            current_max_score = max(current_max_score, score)
            
        # Update best pair representative for box indices
        if current_max_score > merged[key]["best_pair_score"]:
            merged[key]["best_pair_score"] = current_max_score
            merged[key]["best_pair_indices"] = (sbj_idx, obj_idx)
            
    # Format output
    merged_edges = []
    for (sub_gid, obj_gid), data in merged.items():
        # Aggregated no_relation score (mean)
        no_rel_avg = sum(data["no_relation_scores"]) / len(data["no_relation_scores"]) if data["no_relation_scores"] else 0.0
        
        # Convert predicates dict to list of candidates
        final_predicates = []
        for pname, pscore in data["predicates"].items():
            final_predicates.append({
                "predicate": pname,
                "predicate_score": pscore
                # "validated": True
            })
            
        # Sort predicates
        final_predicates.sort(key=lambda x: x['predicate_score'], reverse=True)
        
        # Determine "best" (top 1) compatibility fields
        best_pred = final_predicates[0]['predicate'] if final_predicates else ""
        best_score = final_predicates[0]['predicate_score'] if final_predicates else 0.0
        
        # Construct output compatible with both new list format and old fields
        edge_out = {
            "subject_group_id": sub_gid,
            "object_group_id": obj_gid,
            "no_relation_score": no_rel_avg,
            "predicates": final_predicates, # NEW FIELD
            
            # Old fields for compatibility (populated from best/top 1)
            "best_predicate": best_pred,
            "best_predicate_score": best_score,
            "predicate_scores": {p['predicate']: p['predicate_score'] for p in final_predicates}, # Sparse dict
            
            "subject_box_index": data["best_pair_indices"][0] if data["best_pair_indices"] else None,
            "object_box_index": data["best_pair_indices"][1] if data["best_pair_indices"] else None
        }
        merged_edges.append(edge_out)
        
    return merged_edges


def main():
    ap = argparse.ArgumentParser(
        description="Build anno_merged_edges.{json|pkl} from anno.json, anno_merged.json, and scene-graph.pkl."
    )
    ap.add_argument("--anno", required=True, help="Path to anno.json")
    ap.add_argument("--merged", required=True, help="Path to anno_merged.json")
    ap.add_argument("--scene-pkl", required=True, help="Path to scene-graph.pkl")
    ap.add_argument("--output", required=True, help="Output path (no enforced extension)")
    ap.add_argument(
        "--format",
        choices=["json", "pkl"],
        default="json",
        help="Output format: json (default) or pkl",
    )
    ap.add_argument(
        "--agg",
        choices=["mean", "max"],
        default="mean",
        help="Aggregation method for merging edge scores (mean or max)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="No-relation score threshold for cleaning (default: None)",
    )
    ap.add_argument(
        "--predicate-threshold",
        type=float,
        default=None,
        help="Minimum score for the best predicate to be kept (default: None)",
    )
    ap.add_argument(
        "--relations-json-dir",
        type=str,
        default=None,
        help="Optional directory containing cleaned relations JSON files. If set, overrides pkl relations."
    )
    args = ap.parse_args()

    anno_path = Path(args.anno)
    merged_path = Path(args.merged)
    scene_pkl_path = Path(args.scene_pkl)

    anno_data = json.loads(anno_path.read_text())
    merged_mappings = json.loads(merged_path.read_text())  # {image_id_str: {old_id: new_id}}

    class_map, label_to_cids = build_class_maps(anno_data)
    anno_by_id = {entry["image_id"]: entry for entry in anno_data["data"]}

    # load scene-graph.pkl (list of entries)
    # load scene-graph.pkl (list of entries)
    # Even if we use relations-json-dir, we still need pkl for bboxes/labels mapping consistency check?
    # Yes, we need 'bboxes' and 'box_label' from pkl to build 'boxes' list that matches indices.
    with scene_pkl_path.open("rb") as f:
        scene_entries = pickle.load(f)
        
    # Pre-load JSON relations if available
    json_relations_map = {}
    if args.relations_json_dir:
        json_dir = Path(args.relations_json_dir)
        if json_dir.exists():
            for fpath in json_dir.glob("scene-graph_*.json"):
                # extracting image id from filename scene-graph_123.json
                try:
                    fname = fpath.stem
                    # format: scene-graph_000000001234
                    parts = fname.split('_')
                    if len(parts) >= 2:
                        iid = int(parts[-1])
                        with open(fpath, 'r') as jf:
                            json_relations_map[iid] = json.load(jf)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")
            print(f"Loaded {len(json_relations_map)} JSOn relation files.")

    thing_classes = anno_data["thing_classes"]
    stuff_classes = anno_data["stuff_classes"]
    predicate_classes = anno_data["predicate_classes"]

    label_names = thing_classes + stuff_classes

    out = {}

    for entry in scene_entries:
        img_id = int(entry["img_id"])
        img_id_str = str(img_id)

        if img_id not in anno_by_id:
            continue

        img_anno = anno_by_id[img_id]
        id_mapping = merged_mappings.get(img_id_str, {})

        # 1) merged nodes (groups)
        groups = compute_groups_for_image(img_anno, id_mapping)

        # 2) fill labels
        for gid, g in groups.items():
            t = "thing" if g["isthing"] else "stuff"
            g["label"] = class_map.get((t, g["category_id"]), "")

        # 3) build 'boxes' from pkl (to reuse assign_boxes_to_groups)
        boxes = []
        bboxes = entry["bboxes"]          # (N, 4)
        box_labels = entry["box_label"]   # (N,)
        num_boxes = len(bboxes)
        for idx in range(num_boxes):
            label_idx = int(box_labels[idx])
            label = label_names[label_idx]
            x1, y1, x2, y2 = map(int, bboxes[idx])
            boxes.append({
                "index": idx,
                "label": label,
                "bbox_xyxy": [x1, y1, x2, y2],
            })

        # 4) assign boxes -> groups
        box_to_group = assign_boxes_to_groups(boxes, groups, label_to_cids)

        # 5) build merged edges using full predicate probabilities
        edges = build_merged_edges_from_pkl(
            pairs=entry["pairs"],
            rel_scores=entry["rel_scores"],
            box_to_group=box_to_group,
            predicate_classes=predicate_classes,
            skip_self=True,
            agg=args.agg,
            threshold=args.threshold,
            predicate_threshold=args.predicate_threshold,
        )
        
        # Override with JSON relations if available
        if json_relations_map and img_id in json_relations_map:
            json_data = json_relations_map[img_id]
            # Verify basic consistency?
            # Build merged edges from JSON list
            edges = build_merged_edges_from_json(
                relations_list=json_data.get("relations", []),
                box_to_group=box_to_group,
                skip_self=True
            )

        # 6) Resolve specific segment IDs for granularity
        for edge in edges:
            # Subject
            sbj_idx = edge.get("subject_box_index")
            sbj_gid = edge["subject_group_id"]
            if sbj_idx is not None and sbj_gid in groups:
                 sbj_box = boxes[sbj_idx]["bbox_xyxy"]
                 group_segs = groups[sbj_gid].get("segments_detail", [])
                 
                 # Default to first if list exists
                 best_s_id = group_segs[0]["id"] if group_segs else None
                 best_s_iou = -1.0
                 
                 for s_det in group_segs:
                     iou = iou_xyxy(sbj_box, s_det["bbox"])
                     if iou > best_s_iou:
                         best_s_iou = iou
                         best_s_id = s_det["id"]
                 edge["subject_seg_id"] = best_s_id
            
            # Object
            obj_idx = edge.get("object_box_index")
            obj_gid = edge["object_group_id"]
            if obj_idx is not None and obj_gid in groups:
                 obj_box = boxes[obj_idx]["bbox_xyxy"]
                 group_segs = groups[obj_gid].get("segments_detail", [])
                 
                 best_o_id = group_segs[0]["id"] if group_segs else None
                 best_o_iou = -1.0
                 
                 for s_det in group_segs:
                     iou = iou_xyxy(obj_box, s_det["bbox"])
                     if iou > best_o_iou:
                         best_o_iou = iou
                         best_o_id = s_det["id"]
                 edge["object_seg_id"] = best_o_id

        out[img_id_str] = {
            "image_id": img_id,
            "file_name": img_anno["file_name"],
            "groups": list(groups.values()),
            "edges": edges,
        }

    # Create pickle or json
    if args.format == "json":
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Saved merged graph as JSON to {args.output}")
    else:
        with open(args.output, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved merged graph as pickle to {args.output}")


if __name__ == "__main__":
    main()
