#!/usr/bin/env python3
"""
Image Understanding Evaluation (IGE step)

For each *attribute* or *relationship* question generated from a GT scene graph,
look up the corresponding "answer" from the **predicted** scene graph of a given
model.  The predicted answer is found via the graph-matching node mapping:

    GT node ID  --[graph matching]--> predicted node ID  --> predicted scene graph

Output: a JSONL file in the same format as the VQA answers, where each line
contains   { question_id, question, gt_answer, model_answer, question_type, image_id }.
This JSONL can then be scored by the existing LLM-as-a-Judge (llm_judge.py).

Questions of type  count_objects  and  count_comparison  are now supported.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def build_gt_to_pred_node_map(matched_results: dict, image_id: str, matching_data: dict = None) -> dict:
    """Return a dict  {gt_node_id_str: pred_node_id_str}  for one image.

    Segment-only mode: map strictly by matched node IDs (seg_ids in matching).
    No group or index propagation is performed.
    """
    img_result = matched_results.get("per_image_results", {}).get(image_id)
    if img_result is None:
        return {}

    mapping = {}
    
    # 1. First Pass: Map by explicit ID from matching results
    for pair in img_result.get("matched_node_pairs", []):
        # pair = [pred_id, gt_id, pred_label, gt_label]
        pred_id_str = str(pair[0])
        gt_id_str   = str(pair[1])
        mapping[gt_id_str] = pred_id_str

    return mapping


def load_pred_scene_graph(pred_sg_dir: str, image_id: str) -> dict | None:
    """Load the predicted scene graph for a single image."""
    path = os.path.join(pred_sg_dir, f"scene-graph_{image_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def build_pred_node_index(pred_sg: dict) -> dict:
    """Map  pred_node_id_str -> box_dict  for fast lookup.
    
    Includes both top-level box IDs and sub-member IDs.
    """
    idx = {}
    for box in pred_sg.get("boxes", []):
        # 1. Map by box ID
        idx[str(box["id"])] = box
        
        # 2. Map by sub-member IDs (for multi-segment nodes like 'person')
        for member in box.get("member_attributes", []):
            m_id = str(member.get("seg_id"))
            if m_id and m_id != "None":
                idx[m_id] = box
    return idx


def build_pred_edge_index(pred_sg_for_matching: dict, image_id: str) -> dict:
    """
    Build an edge lookup from scene_graphs_for_matching.json.
    Returns { (src_id_str, tgt_id_str): [predicate, ...] }
    """
    images = pred_sg_for_matching.get("images", {})
    img_data = images.get(image_id, {})
    pred_data = img_data.get("prediction", {})
    edge_map = defaultdict(list)
    for edge in pred_data.get("edges", []):
        src = str(edge["source"])
        tgt = str(edge["target"])
        rel = edge.get("relation") or edge.get("label") or edge.get("predicate", "unknown")
        edge_map[(src, tgt)].append(rel)
    return dict(edge_map)


# ---------------------------------------------------------------------------
# Answer-generation logic
# ---------------------------------------------------------------------------

def answer_attribute_question(question: dict,
                              gt_to_pred: dict,
                              pred_node_index: dict) -> tuple:
    """
    For a  label_to_attribute  question, look up the attribute value on the
    *matched* predicted node.
    
    Returns (answer_str, debug_dict).
    """
    ref = question.get("reference") or question.get("meta") or {}
    gt_obj_id = str(ref.get("object_id") or ref.get("object_idx", ""))
    attr_key = ref.get("attribute_key") or ref.get("attribute", "")

    debug = {
        "gt_obj_id_used": gt_obj_id,
        "pred_matched_seg_id": None,
        "pred_attribute_key": attr_key,
        "pred_attribute_value": None,
        "lookup_path": None,
    }

    pred_id = gt_to_pred.get(gt_obj_id)
    if pred_id is None:
        debug["lookup_path"] = "no_match_in_gt_to_pred"
        return "None.", debug

    debug["pred_matched_seg_id"] = pred_id

    pred_box = pred_node_index.get(pred_id)
    if pred_box is None:
        debug["lookup_path"] = "pred_id_not_in_node_index"
        return "None.", debug

    # --- Attribute Lookup Logic (Strict Segment Mode) ---
    attrs = None
    # Check if the matched ID is a member_attribute ID
    for member in pred_box.get("member_attributes", []):
        if str(member.get("seg_id")) == pred_id:
            attrs = member.get("attributes", {})
            debug["lookup_path"] = "member_attributes"
            break
    
    # Fallback: check top-level attributes (only if the box id matches EXACTLY)
    if attrs is None and str(pred_box.get("id")) == pred_id:
        attrs = pred_box.get("attributes", {})
        debug["lookup_path"] = "direct_attributes"

    if attrs is None:
        debug["lookup_path"] = "no_attributes_found_for_id"
        return "None.", debug

    val = attrs.get(attr_key)
    if val is None:
        debug["lookup_path"] = f"attr_key_{attr_key}_not_found"
        debug["pred_available_keys"] = list(attrs.keys()) if attrs else []
        return "None.", debug
    
    debug["pred_attribute_value"] = val
    if isinstance(val, list):
        return ", ".join(str(v) for v in val), debug
    return str(val), debug


def answer_attributes_to_label_question(question: dict,
                                        gt_to_pred: dict,
                                        pred_node_index: dict) -> tuple:
    """
    For an  attributes_to_label  question the answer is the *predicted* label
    of the matched node.
    
    Returns (answer_str, debug_dict).
    """
    ref = question.get("reference") or question.get("meta") or {}
    gt_obj_id = str(ref.get("object_id") or ref.get("object_idx", ""))

    debug = {
        "gt_obj_id_used": gt_obj_id,
        "pred_matched_seg_id": None,
        "pred_label": None,
        "lookup_path": None,
    }

    pred_id = gt_to_pred.get(gt_obj_id)
    if pred_id is None:
        debug["lookup_path"] = "no_match_in_gt_to_pred"
        return "None.", debug

    debug["pred_matched_seg_id"] = pred_id

    pred_box = pred_node_index.get(pred_id)
    if pred_box is None:
        debug["lookup_path"] = "pred_id_not_in_node_index"
        return "None.", debug

    label = pred_box.get("label", "None.")
    debug["pred_label"] = label
    debug["lookup_path"] = "success"
    return label, debug


def answer_relationship_question(question: dict,
                                 gt_to_pred: dict,
                                 pred_edge_index: dict) -> tuple:
    """
    For a  label_attributes_to_relationship  question, look up the edge
    between the two *matched* predicted nodes.
    
    Returns (answer_str, debug_dict).
    """
    ref = question.get("reference") or question.get("meta") or {}
    gt_subj_id = str(ref.get("subject_id") or ref.get("subject_idx", ""))
    gt_obj_id = str(ref.get("object_id") or ref.get("object_idx", ""))

    debug = {
        "gt_subject_id_used": gt_subj_id,
        "gt_object_id_used": gt_obj_id,
        "pred_subject_seg_id": None,
        "pred_object_seg_id": None,
        "pred_predicates": None,
        "lookup_path": None,
    }

    pred_subj = gt_to_pred.get(gt_subj_id)
    pred_obj = gt_to_pred.get(gt_obj_id)

    debug["pred_subject_seg_id"] = pred_subj
    debug["pred_object_seg_id"] = pred_obj

    if pred_subj is None or pred_obj is None:
        missing = []
        if pred_subj is None:
            missing.append("subject")
        if pred_obj is None:
            missing.append("object")
        debug["lookup_path"] = f"no_match_for_{'+'.join(missing)}"
        return "None.", debug

    rels = pred_edge_index.get((pred_subj, pred_obj), [])
    debug["pred_predicates"] = rels if rels else None
    if not rels:
        debug["lookup_path"] = "no_edge_between_matched_nodes"
        return "None.", debug
    debug["lookup_path"] = "success"
    if len(rels) == 1:
        return rels[0], debug
    elif len(rels) == 2:
        return f"{rels[0]} and {rels[1]}", debug
    else:
        return ", ".join(rels[:-1]) + f", and {rels[-1]}", debug


# Counting questions have been removed per user request.

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 'answers' from predicted scene graphs for consistency evaluation."
    )
    parser.add_argument("--questions", required=True,
                        help="Path to generated_questions (or sampled subset) JSON file.")
    parser.add_argument("--matched-graphs", required=True,
                        help="Path to scene_graphs_matched.json for the target model.")
    parser.add_argument("--matching-graphs", required=True,
                        help="Path to scene_graphs_for_matching.json (contains edges).")
    parser.add_argument("--pred-sg-dir", required=True,
                        help="Directory containing scene-graph_<image_id>.json predicted files (5_attributes_pt/<model>).")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path (same format as VQA answer files).")
    args = parser.parse_args()

    # ---- load questions ----
    print(f"Loading questions from {args.questions}")
    with open(args.questions, "r") as f:
        all_questions = json.load(f)["questions"]

    # Filter to relevant questions, mapping back to their original IDs
    relevant_types = {
        "label_to_attribute", 
        "attributes_to_label", 
        "label_attributes_to_relationship"
    }
    
    # Pre-index to avoid O(N^2) lookups later
    questions_with_ids = [
        (qi, q) for qi, q in enumerate(all_questions) 
        if q.get("question_type") in relevant_types
    ]
    
    print(f"  Total questions: {len(all_questions)},  Relevant: {len(questions_with_ids)}")

    # ---- load matched graphs ----
    print(f"Loading matched graphs from {args.matched_graphs}")
    with open(args.matched_graphs, "r") as f:
        matched_data = json.load(f)

    # ---- load matching graphs (for edges) ----
    print(f"Loading matching graphs from {args.matching_graphs}")
    with open(args.matching_graphs, "r") as f:
        matching_data = json.load(f)

    # ---- process each question ----
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    stats = defaultdict(int)
    records = []

    # Cache per-image data
    _gt_to_pred_cache = {}
    _pred_node_cache = {}
    _pred_edge_cache = {}

    for qi, q in questions_with_ids:
        image_id = str(q["image_id"])
        qtype = q["question_type"]

        # Build caches lazily
        if image_id not in _gt_to_pred_cache:
            _gt_to_pred_cache[image_id] = build_gt_to_pred_node_map(matched_data, image_id, matching_data)

            pred_sg = load_pred_scene_graph(args.pred_sg_dir, image_id)
            if pred_sg is not None:
                _pred_node_cache[image_id] = build_pred_node_index(pred_sg)
            else:
                _pred_node_cache[image_id] = {}

            _pred_edge_cache[image_id] = build_pred_edge_index(matching_data, image_id)

        gt_to_pred = _gt_to_pred_cache[image_id]
        pred_nodes = _pred_node_cache[image_id]
        pred_edges = _pred_edge_cache[image_id]

        # Generate answers
        if qtype == "label_to_attribute":
            model_answer, answer_debug = answer_attribute_question(q, gt_to_pred, pred_nodes)
        elif qtype == "attributes_to_label":
            model_answer, answer_debug = answer_attributes_to_label_question(q, gt_to_pred, pred_nodes)
        elif qtype == "label_attributes_to_relationship":
            model_answer, answer_debug = answer_relationship_question(q, gt_to_pred, pred_edges)
        else:
            continue

        stats[qtype] += 1
        if model_answer == "None.":
            stats[f"{qtype}_unmatched"] += 1

        # The original_qid is strictly tracked
        original_qid = qi

        # Merge question-level debug info (from generation) with answer-level debug info
        q_debug = q.get("debug", {})
        combined_debug = {**q_debug, **answer_debug}

        records.append({
            "question_id": original_qid,
            "question_type": qtype,
            "question": q["question"],
            "gt_answer": q["answer"],
            "model_answer": model_answer,
            "image_id": image_id,
            "image_name": f"{image_id}.jpg",
            "reference": q.get("reference", {}),
            "debug": combined_debug,
        })

    # ---- write output ----
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(records)} IGE records to {args.output}")
    print(f"\nBreakdown:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
