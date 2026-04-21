#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import random
import pickle
import numpy as np
import subprocess
from itertools import combinations
from typing import List, Dict


# Ensure vqa_utils is importable
try:
    from vqa_utils import AttributeFormatter, ObjectDescriber, LabelFormatter, RelationFormatter
    _VQA_UTILS = True
except ImportError:
    AttributeFormatter = None
    ObjectDescriber = None
    LabelFormatter = None
    RelationFormatter = None
    _VQA_UTILS = False

from scene_struct import build_scene_struct


def _trace_meta(obj: dict, obj_idx: int) -> dict:
    """
    Traceback pointer for a scene_struct object.
    Segment-only: seg_id is the canonical instance identifier.
    """
    return {
        "object_idx": obj_idx,                       # index in scene_struct["objects"]
        "label": obj.get("label"),
        "seg_id": obj.get("seg_id"),                 # present for member-expanded objects
        "box_id": obj.get("box_id"),                 # present if added to scene_struct
        "box_index": obj.get("original_index"),      # fallback (source box index)
        "is_member": obj.get("is_member", False),
        "bbox": obj.get("bbox"),
    }


def _is_empty_attr_value(v) -> bool:
    """Check if attribute value is effectively empty."""
    if v in (None, "", [], {}):
        return True
    if isinstance(v, list):
        # Check if list contains only empty values
        return all(str(x).strip() == "" for x in v)
    if isinstance(v, str):
        return not v.strip()
    return False

def _clean_label(label: str) -> str:
    """Clean label for display."""
    if _VQA_UTILS and LabelFormatter:
        return LabelFormatter.format_label(label)
    return str(label).replace("-merged", "").replace("-other", "").replace("-", " ").strip()

def _clean_relation(rel: str) -> str:
    """Clean relationship predicate."""
    if _VQA_UTILS and RelationFormatter:
        return RelationFormatter.format_relation(rel)
    return str(rel).strip()

def _format_value(v) -> str:
    """Format attribute value for answer/description."""
    if _VQA_UTILS and AttributeFormatter:
        return AttributeFormatter.format_value(v)
    
    # Fallback implementation
    if _is_empty_attr_value(v):
        return ""
    if isinstance(v, list):
        if not v:
            return ""
        if len(v) == 1:
            return str(v[0])
        elif len(v) == 2:
            return f"{v[0]} and {v[1]}"
        else:
            return ", ".join(map(str, v[:-1])) + f", and {v[-1]}"
    return str(v)

def _format_key(k: str) -> str:
    """Format attribute key for question."""
    if _VQA_UTILS and AttributeFormatter:
        return AttributeFormatter.format_key(k)
    return k.replace("_", " ")

def _stem_simple(word: str) -> str:
    """Very simple stemming to catch 'cleaner' vs 'cleaning' overlap."""
    word = word.lower().strip()
    # Sort suffixes by length to match longest first
    for suffix in ["ing", "er", "ed", "es", "s"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

def _value_equal_simple(v1, v2, synonyms=None) -> bool:
    """Equality check for attribute values with overlap checking.
    
    Returns True if values are considered 'conflicting' or 'overlapping'
    such that one cannot clearly distinguish from the other.
    """
    if _is_empty_attr_value(v1):
        return _is_empty_attr_value(v2)
    if _is_empty_attr_value(v2):
        return False

    a = v1 if isinstance(v1, list) else [v1]
    b = v2 if isinstance(v2, list) else [v2]

    # Normalize to sets of lower-case strings
    a_strs = {str(x).strip().lower() for x in a if str(x).strip()}
    b_strs = {str(x).strip().lower() for x in b if str(x).strip()}
    
    # 1. Direct intersection
    if len(a_strs & b_strs) > 0:
        return True
    
    # 2. Substring matching (weak semantic overlap in raw strings)
    # "shirt" in "t-shirt"
    for x in a_strs:
        for y in b_strs:
            if x in y or y in x:
                return True

    # 3. Stemming overlap (e.g. "cleaner" vs "cleaning product")
    # Tokenize and stem
    def get_stems(strs):
        stems = set()
        for s in strs:
            for token in s.split():
                stems.add(_stem_simple(token))
        return stems

    a_stems = get_stems(a_strs)
    b_stems = get_stems(b_strs)
    
    # Require non-trivial overlap (ignore common stop words if we had them, but here we assume attrs are meaningful)
    # If they share ANY stem, they are confusing.
    if len(a_stems & b_stems) > 0:
        return True
                
    # 4. Synonyms (if provided)
    if synonyms:
        pass
        
    return False

def _pluralize(label: str) -> str:
    """Simple pluralizer."""
    if label.endswith("s") or label.endswith("sh") or label.endswith("ch") or label.endswith("x") or label.endswith("z"):
        return label + "es"
    if label.endswith("y") and len(label) > 1 and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    return label + "s"

def _find_min_unique_attr_keys_same_label(scene_struct, obj_idx: int, allowed_keys: list, synonyms=None) -> list:
    """Find minimal set of attributes to distinguish object from others with SAME label."""
    objects = scene_struct["objects"]
    target = objects[obj_idx]
    label = target.get("label")
    target_attrs = target.get("attrs", {}) or {}

    competitors = [o for i, o in enumerate(objects) if i != obj_idx and o.get("label") == label]
    if not competitors:
        return []

    # Valid candidate keys that actually exist on target
    candidate_keys = [k for k in allowed_keys if k in target_attrs and not _is_empty_attr_value(target_attrs.get(k))]
    if not candidate_keys:
        return []

    # Try to find minimal combination
    for r in range(1, len(candidate_keys) + 1):
        for combo in combinations(candidate_keys, r):
            unique = True
            for comp in competitors:
                comp_attrs = comp.get("attrs", {}) or {}
                # Two objects are effectively same if ALL attributes in combo match/overlap
                if all(_value_equal_simple(target_attrs.get(k), comp_attrs.get(k), synonyms) for k in combo):
                    unique = False
                    break
            if unique:
                return list(combo)

    return []

def _find_min_unique_attr_keys_global(scene_struct, obj_idx: int, allowed_keys: list, synonyms=None) -> list:
    """Find minimal set of attributes to distinguish object from ALL other objects."""
    objects = scene_struct["objects"]
    target = objects[obj_idx]
    target_attrs = target.get("attrs", {}) or {}

    competitors = [o for i, o in enumerate(objects) if i != obj_idx]
    if not competitors:
        return []

    candidate_keys = [k for k in allowed_keys if k in target_attrs and not _is_empty_attr_value(target_attrs.get(k))]
    if not candidate_keys:
        return []

    for r in range(1, len(candidate_keys) + 1):
        for combo in combinations(candidate_keys, r):
            unique = True
            for comp in competitors:
                comp_attrs = comp.get("attrs", {}) or {}
                # Check uniqueness against this competitor
                if all(_value_equal_simple(target_attrs.get(k), comp_attrs.get(k), synonyms) for k in combo):
                    unique = False
                    break
            if unique:
                return list(combo)

    return []

def _describe_object_required(scene_struct, obj_idx: int, attr_keys: list, include_label: bool = True) -> str:
    """Describe object using specific attribute keys + proper formatting."""
    obj = scene_struct["objects"][obj_idx]
    label = _clean_label(obj.get("label", "object")) if include_label else ""
    attrs = obj.get("attrs", {}) or {}
    
    # Extract subset of attributes needed for description
    subset = {k: attrs.get(k) for k in attr_keys if k in attrs and not _is_empty_attr_value(attrs.get(k))}

    if _VQA_UTILS and ObjectDescriber is not None:
        # Use utility for natural phrasing (e.g. putting color before noun)
        desc_obj = {"label": label, "attrs": subset}
        return " ".join(ObjectDescriber.describe_single_object(desc_obj, include_attrs=True).split())

    # Fallback description logic
    parts = []
    # Simple strategy: adjectives first if known, else post-nominal? 
    # For now, just listing them.
    for k in attr_keys:
        if k in subset:
            val_str = _format_value(subset[k])
            parts.append(val_str)
    
    if label:
        parts.append(label)
        
    return " ".join(parts) if parts else (label or "object")

def generate_required_vqa_questions(
    scene_struct, 
    metadata, 
    synonyms=None, 
    all_predicates=None, 
    use_multiple_choice=False,
    scene_graph_id: str = "unknown"
) -> List[Dict]:
    """Generate the 3 required question types with uniqueness constraints.

    1. Label -> Attribute (Ask about attribute of a specific object)
    2. Attributes -> Label (Identify object from attributes)
    3. Label + Attributes -> Relationship (Relationship between two objects)
    """
    objects = scene_struct.get("objects", [])
    rels = scene_struct.get("relationships", {}) or {}
    rel_scores = scene_struct.get("relationship_scores", {}) or {}
    
    # IGNORE metadata filtering. Allow all attributes present in the graph.
    # We rely on _is_empty_attr_value to filter bad values.
    # allowed_attr_keys = metadata.get("types", {}).get("AttributeType", [])
    
    label_counts = {}
    for obj in objects:
        lbl = obj.get("label")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    out: List[Dict] = []

    # Get all potential attribute keys present in this scene to use for disambiguation
    # logic if needed, although we usually just look at the object's own attributes.
    
    # --- Type A: Ask for Attribute (Given Label [+ Disambiguating Attributes]) ---
    # "What is the <attr> of the <object>?"
    for obj_idx, obj in enumerate(objects):
        label = obj.get("label")
        attrs = obj.get("attrs", {}) or {}

        # Sort keys for deterministic output
        for attr_key, raw_val in sorted(attrs.items(), key=lambda kv: str(kv[0])):
            # Skip if value is empty
            if _is_empty_attr_value(raw_val):
                continue
            
            # Use AttributeFormatter to check if this is a "meaningful" attribute
            # If we have a formatter, format_key usually maps known good attributes to strings.
            # If it returns the key itself (and keys have underscore), it might be technical.
            # But let's be permissible.
            
            # Answer
            answer = _format_value(raw_val)
            if not answer.strip():
                continue

            # Need to describe the object uniquely WITHOUT using the target attribute
            # We can use ANY other attribute present on the object.
            disambig_allowed = [k for k in attrs.keys() if k != attr_key and not _is_empty_attr_value(attrs.get(k))]
            
            disambig_keys = []
            if label_counts.get(label, 0) > 1:
                disambig_keys = _find_min_unique_attr_keys_same_label(scene_struct, obj_idx, disambig_allowed, synonyms)
                # If we cannot uniquely identify it, we strictly cannot ask about a specific instance's attribute
                # unless we are ok with ambiguity (user says "target unique instances").
                if not disambig_keys:
                    # Alternative: If all instances of 'label' have the SAME value for 'attr_key',
                    # the question "What is the X of the label?" is valid even if instances are not distinguished.
                    # But the prompt implies unique instances. We'll skip if ambiguous.
                    continue
            
            obj_desc = _describe_object_required(scene_struct, obj_idx, disambig_keys, include_label=True)
            attr_text = _format_key(attr_key)
            
            # Avoid questions like "What is the color of the red car?" (if answer is red)
            # But here attr_text is the KEY (e.g. "color"), answer is "red".
            # obj_desc might contain "red" if needed for disambiguation.
            # "What is the color of the red car?" Answer: "red". This is tautological.
            # We should check if the answer is contained in the description?
            # Or reliance on _find_min_unique_attr_keys ensuring we pick attributes DIFFERENT from the one we ask?
            # Yes, disambig_allowed excludes attr_key.
            # BUT if attr_key is "primary_color" and disambig used "color" (if such alias existed)...
            # Assuming keys are distinct properties.
            
            question = f"What is the {attr_text} of the {obj_desc}?"
            
            # Program trace
            program = [
                {"type": "scene", "inputs": []},
                {"type": "filter_index", "inputs": [0], "side_inputs": [str(obj_idx)]},
                {"type": "unique", "inputs": [1]},
                {"type": "query_attr", "inputs": [2], "side_inputs": [attr_key]},
            ]

            out.append({
                "question_type": "label_to_attribute",
                "question": question,
                "program": program,
                "answer": answer,
                "reference": {
                    "type": "attribute",
                    "scene_graph": scene_graph_id,
                    "object_id": obj.get("seg_id"),
                    "attribute_key": attr_key
                },
                "meta": {
                    **_trace_meta(obj, obj_idx),
                    "attribute": attr_key
                },
                "debug": {
                    "gt_seg_id": obj.get("seg_id"),
                    "gt_object_label": obj.get("label"),
                    "gt_object_idx": obj_idx,
                    "gt_attribute_key": attr_key,
                    "gt_attribute_value": raw_val,
                },
            })

    # --- Type B: Ask for Label (Given Globally Unique Attributes) ---
    # "What object is <attrs>?"
    for obj_idx, obj in enumerate(objects):
        # Allow all attributes for unique identification
        allowed_keys = [k for k in obj.get("attrs", {}).keys() if not _is_empty_attr_value(obj["attrs"][k])]
        unique_keys = _find_min_unique_attr_keys_global(scene_struct, obj_idx, allowed_keys, synonyms)
        if not unique_keys:
            continue

        # Answer
        answer = _clean_label(obj.get("label", ""))
        if not answer.strip():
            continue

        # Describe object using ONLY attributes (no label)
        attrs = obj.get("attrs", {})
        
        if _VQA_UTILS and AttributeFormatter:
             parts = []
             for k in unique_keys:
                 v = attrs.get(k)
                 phrase = AttributeFormatter.format_attr_phrase(k, v)
                 if phrase:
                     parts.append(phrase)
             
             if not parts:
                 # Fallback if formatter returns empty (unlikely)
                 parts = [_format_value(attrs.get(k)) for k in unique_keys]
             
             if len(parts) == 1:
                 attrs_desc = parts[0]
             elif len(parts) == 2:
                 attrs_desc = f"{parts[0]} and {parts[1]}"
             else:
                 attrs_desc = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        else:
             parts = [_format_value(attrs.get(k)) for k in unique_keys]
             attrs_desc = ", ".join(parts)

        question = f"What object is {attrs_desc}?"
        # Handle "with" phrases better
        question = question.replace(" is with ", " has ")
        question = question.replace(", with ", ", and has ")
        question = question.replace(" and with ", " and has ")

        # Further cleanup for "is has" edge case if any (unlikely with replace order but safe)
        question = question.replace("is has ", "has ")

        
        nodes = [{"type": "scene", "inputs": []}]
        cur = 0
        for k in unique_keys:
            v_raw = attrs.get(k)
            val_prog = v_raw[0] if isinstance(v_raw, list) and v_raw else v_raw
            nodes.append({"type": "filter_attr", "inputs": [cur], "side_inputs": [k, str(val_prog)]})
            cur += 1
        nodes.append({"type": "unique", "inputs": [cur]})
        nodes.append({"type": "query_label", "inputs": [cur + 1]})

        out.append({
            "question_type": "attributes_to_label",
            "question": question,
            "program": nodes,
            "answer": answer,
            "reference": {
                "type": "attribute_to_label",
                "scene_graph": scene_graph_id,
                    "object_id": obj.get("seg_id"),
                "attribute_keys": unique_keys
            },
            "meta": {
                **_trace_meta(obj, obj_idx),
                "unique_attribute_keys": unique_keys
            },
            "debug": {
                    "gt_seg_id": obj.get("seg_id"),
                "gt_object_label": obj.get("label"),
                "gt_object_idx": obj_idx,
                "gt_attribute_keys": unique_keys,
                "gt_attribute_values": {k: attrs.get(k) for k in unique_keys},
            },
        })

    # --- Type C: Relationship (Label+Attrs -> Relation -> Label+Attrs) ---
    # "What is the relationship between the <subj> and the <obj>?"

    # We want at most one question per unique (subject_idx, object_idx) pair.
    # But we want to capture ALL valid relationships in the ANSWER.
    pair_relations = {} # (subj, obj) -> set(rel_names)

    # First pass: collect all relationships for each pair
    for rel_name, mapping in rels.items():
        for subj_idx, obj_indices in mapping.items():
            subj_i = int(subj_idx)
            for obj_i in obj_indices:
                if subj_i == obj_i: continue
                pair = (subj_i, obj_i)
                if pair not in pair_relations:
                    pair_relations[pair] = set()
                pair_relations[pair].add(_clean_relation(rel_name))

    # Second pass: generate questions
    processed_pairs = set()

    for rel_name, mapping in sorted(rels.items(), key=lambda kv: str(kv[0])):
        for subj_idx_raw, obj_indices in sorted(mapping.items(), key=lambda kv: int(kv[0])):
            subj_idx = int(subj_idx_raw)
            for obj_idx in obj_indices:
                if (subj_idx, obj_idx) in processed_pairs:
                    continue

                subj = objects[subj_idx]
                obj = objects[obj_idx]
                
                subj_label = subj.get("label")
                obj_label = obj.get("label")

                # Disambiguate Subject
                subj_keys = []
                if label_counts.get(subj_label, 0) > 1:
                    allowed_s = [k for k in subj.get("attrs", {}).keys() if not _is_empty_attr_value(subj["attrs"][k])]
                    subj_keys = _find_min_unique_attr_keys_same_label(scene_struct, subj_idx, allowed_s, synonyms)
                    if not subj_keys:
                        continue

                # Disambiguate Object
                obj_keys = []
                if label_counts.get(obj_label, 0) > 1:
                    allowed_o = [k for k in obj.get("attrs", {}).keys() if not _is_empty_attr_value(obj["attrs"][k])]
                    obj_keys = _find_min_unique_attr_keys_same_label(scene_struct, obj_idx, allowed_o, synonyms)
                    if not obj_keys:
                        continue

                subj_desc = _describe_object_required(scene_struct, subj_idx, subj_keys, include_label=True)
                obj_desc = _describe_object_required(scene_struct, obj_idx, obj_keys, include_label=True)

                # Combine all relationships into the answer
                all_rels = sorted(list(pair_relations.get((subj_idx, obj_idx), [])))
                if not all_rels:
                    continue
                
                if len(all_rels) == 1:
                    answer = all_rels[0]
                elif len(all_rels) == 2:
                    answer = f"{all_rels[0]} and {all_rels[1]}"
                else:
                    answer = ", ".join(all_rels[:-1]) + f", and {all_rels[-1]}"

                # Determine question template based on predicates
                # Determine question template based on predicates
                # Predicate Groups
                groups = {
                    "Spatial": [
                        'over', 'above', 'in front of', 'beside', 'next to', 
                        'on', 'atop', 'in', 'inside', 'attached to', 
                        'hanging from', 'on back of', 'painted on', 'parked on', 
                        'enclosing', 'covering', 'leaning on'
                    ],
                    "Posture": [
                        'standing on', 'sitting on', 'lying on', 'kneeling on', 
                        'perched on', 'wearing', 'dressed in'
                    ],
                    "Locomotion": [
                        'walking on', 'running on', 'jumping over', 'jumping from', 
                        'flying over', 'falling off', 'going down', 'crossing', 
                        'climbing', 'entering', 'exiting', 'driving on', 'riding', 
                        'chasing'
                    ],
                    "Interaction": [
                        'holding', 'carrying', 'picking', 'picking up', 'touching', 'pushing', 
                        'pulling', 'opening', 'closing', 'throwing', 'catching', 
                        'slicing', 'cutting', 'cooking', 'cleaning', 'driving', 'guiding', 
                        'swinging', 'kicking', 'eating', 'drinking'
                    ],
                    "Social": [
                        'looking at', 'watching', 'talking to', 'listening to', 
                        'kissing', 'hugging', 'feeding', 'playing with', 
                        'playing', 'fighting', 'hitting', 'biting', 'smiling at', 'about to hit'
                    ]
                }
                
                # Reverse mapping for easy lookup
                pred_to_group = {}
                for g, preds in groups.items():
                    for p in preds:
                        pred_to_group[p] = g
                
                templates = {
                    "Spatial": "What is the spatial position of the {subj} relative to the {obj}?",
                    "Posture": "What is the physical pose or configuration of the {subj} relative to the {obj}?",
                    "Locomotion": "How is the {subj} moving through space relative to the {obj}?",
                    "Interaction": "What action is the {subj} performing on the {obj}?",
                    "Social": "How is the {subj} socially or visually engaging with the {obj}?",
                    "Default": "What is the relationship between the {subj} and the {obj}?"
                }

                primary_rel = all_rels[0] 
                group = pred_to_group.get(primary_rel, "Default")
                
                template_str = templates.get(group, templates["Default"])
                
                question = template_str.format(subj=subj_desc, obj=obj_desc)
                
                program = [
                    {"type": "scene", "inputs": []},
                    {"type": "filter_index", "inputs": [0], "side_inputs": [str(subj_idx)]},
                    {"type": "unique", "inputs": [1]},
                    {"type": "filter_index", "inputs": [0], "side_inputs": [str(obj_idx)]},
                    {"type": "unique", "inputs": [3]},
                    {"type": "query_relation", "inputs": [2, 4]},
                ]

                if use_multiple_choice and all_predicates:
                    # Find highest rated relation
                    best_rel = None
                    
                    # Iterate rel_scores for this pair to find best relation
                    candidates = []
                    for (s, o, raw_pred), score in rel_scores.items():
                        if s == subj_idx and o == obj_idx:
                            candidates.append((raw_pred, score))
                    
                    if not candidates:
                         # Fallback if no scores found use first available
                         best_rel = all_rels[0] # all_rels is sorted list of clean names
                    else:
                        # Sort by score descending
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        best_rel_raw = candidates[0][0]
                        best_rel = _clean_relation(best_rel_raw)

                    target = best_rel
                    
                    # Distractors must NOT be in the set of true relations for this pair
                    true_rels_cleaned = set(all_rels)
                    
                    possible_distractors = [p for p in all_predicates if _clean_label(p) not in true_rels_cleaned and _clean_relation(p) not in true_rels_cleaned]
                    
                    if len(possible_distractors) < 3:
                        possible_distractors = list(all_predicates)
                    
                    distractors = random.sample(possible_distractors, min(3, len(possible_distractors)))
                    distractors_clean = [_clean_relation(d) for d in distractors]
                    
                    options = [target] + distractors_clean
                    random.shuffle(options)
                    
                    letters = ["A", "B", "C", "D"]
                    correct_idx = options.index(target)
                    correct_letter = letters[correct_idx]
                    
                    opts_str = ", ".join([f"{l}) {opt}" for l, opt in zip(letters, options)])
                    
                    question = f"{template_str.format(subj=subj_desc, obj=obj_desc)}\n{opts_str}"
                    answer = correct_letter
                    
                    out.append({
                        "question_type": "label_attributes_to_relationship",
                        "question": question,
                        "program": program,
                        "answer": answer,
                        "reference": {
                            "type": "relationship",
                            "scene_graph": scene_graph_id,
                            "subject_id": objects[subj_idx].get("seg_id"),
                            "object_id": objects[obj_idx].get("seg_id"),
                            "predicates": all_rels
                        },
                        "meta": {
                            "subject": _trace_meta(objects[subj_idx], subj_idx),
                            "object": _trace_meta(objects[obj_idx], obj_idx),
                            "predicates": all_rels,
                            "options": options,
                            "correct_option": target
                        },
                        "debug": {
                            "gt_subject_seg_id": objects[subj_idx].get("seg_id"),
                            "gt_subject_label": objects[subj_idx].get("label"),
                            "gt_object_seg_id": objects[obj_idx].get("seg_id"),
                            "gt_object_label": objects[obj_idx].get("label"),
                            "gt_predicates": all_rels,
                        },
                    })

                else:
                    if len(all_rels) == 1:
                        answer = all_rels[0]
                    elif len(all_rels) == 2:
                        answer = f"{all_rels[0]} and {all_rels[1]}"
                    else:
                        answer = ", ".join(all_rels[:-1]) + f", and {all_rels[-1]}"

                    out.append({
                        "question_type": "label_attributes_to_relationship",
                        "question": question,
                        "program": program,
                        "answer": answer,
                        "reference": {
                            "type": "relationship",
                            "scene_graph": scene_graph_id,
                            "subject_id": objects[subj_idx].get("seg_id"),
                            "object_id": objects[obj_idx].get("seg_id"),
                            "predicates": all_rels
                        },
                        "meta": {
                            "subject": _trace_meta(objects[subj_idx], subj_idx),
                            "object": _trace_meta(objects[obj_idx], obj_idx),
                            "predicates": all_rels
                        },
                        "debug": {
                            "gt_subject_seg_id": objects[subj_idx].get("seg_id"),
                            "gt_subject_label": objects[subj_idx].get("label"),
                            "gt_object_seg_id": objects[obj_idx].get("seg_id"),
                            "gt_object_label": objects[obj_idx].get("label"),
                            "gt_predicates": all_rels,
                        },
                    })
                
                processed_pairs.add((subj_idx, obj_idx))

    # Final cleanup
    for q in out:
        # Cleanup extra spaces
        q["question"] = re.sub(r"\s+", " ", q["question"]).strip()
        if not q["question"].endswith("?"):
            q["question"] += "?"
            
    return out


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_raw_scenes(path: str):
    if os.path.isdir(path):
        items = []
        for fp in sorted(glob.glob(os.path.join(path, "scene-graph_*.json"))):
            scene = load_json(fp)
            if isinstance(scene, dict):
                img_id = str(scene.get("image_id", os.path.basename(fp)))
                items.append((img_id, scene))
        if not items:
            # Fallback
            for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
                 if "scene-graph" in fp or "attributes" not in fp: 
                    try:
                        scene = load_json(fp)
                        if isinstance(scene, dict) and ("boxes" in scene or "groups" in scene):
                            img_id = str(scene.get("image_id", os.path.basename(fp)))
                            items.append((img_id, scene))
                        elif isinstance(scene, dict):
                            for k, v in scene.items():
                                if isinstance(v, dict) and ("boxes" in v or "groups" in v):
                                    items.append((str(v.get("image_id", k)), v))
                    except: pass
        return items

    raw = load_json(path)
    if isinstance(raw, dict) and ("groups" in raw or "boxes" in raw):
        return [(str(raw.get("image_id", os.path.basename(path))), raw)]
    if isinstance(raw, dict):
        return [(str(k), v) for k, v in raw.items() if isinstance(v, dict) and ("boxes" in v or "groups" in v)]
    if isinstance(raw, dict):
        return [(str(k), v) for k, v in raw.items() if isinstance(v, dict) and ("boxes" in v or "groups" in v)]
    return []

def load_templates(template_dir: str):
    templates = []
    # Simplified loader for legacy support
    if not os.path.exists(template_dir):
        return []
    for fn in os.listdir(template_dir):
        if fn.endswith(".json"):
            try:
                data = load_json(os.path.join(template_dir, fn))
                if isinstance(data, list): templates.extend(data)
                elif isinstance(data, dict): templates.append(data)
            except: pass
    return templates



def load_merged_edges_scenes(path: str):
    """Load scenes from anno_merged_edges.json with thresholding."""
    with open(path, "r") as f:
        data = json.load(f)
        
    REL_THRESHOLD_FACTOR = 0.8
    items = []
    
    # sort by image_id for determinism
    for img_id_str, entry in sorted(data.items(), key=lambda x: int(x[0])):
        # Convert merged groups to boxes format expected by build_scene_struct
        boxes = []
        for g in entry["groups"]:
            boxes.append({
                "index": g["group_id"], # Use group_id as the primary index
                "label": g["label"],
                "bbox_xyxy": g["bbox"],
                "attributes": {}, # Attributes would come from prop_vectors if available
            })
            
        relations = []
        for edge in entry["edges"]:
            scores = edge["predicate_scores"]
            no_rel = edge["no_relation_score"]
            
            # scores is dict {predicate: score}
            # Find max score among predicates
            if not scores:
                continue
                
            sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            max_score = sorted_preds[0][1]
            threshold = max_score * REL_THRESHOLD_FACTOR
            
            for pred, score in sorted_preds:
                if score < threshold:
                    break
                
                relations.append({
                    "subject_index": edge["subject_group_id"],
                    "object_index": edge["object_group_id"],
                    "predicate": pred,
                    "predicate_score": score,
                    "no_relation_score": no_rel
                })
                
        items.append((img_id_str, {
            "image_id": entry["image_id"],
            "file_name": entry["file_name"],
            "boxes": boxes,
            "relations": relations
        }))
        
    return items


def load_validated_scenes(dir_path: str):
    """Load scenes from VLM-validated clean_and_refine output directory.
    
    Reads per-image scene-graph_*.json files from the clean_and_refine step.
    All relations in these files have been validated by a VLM, so they are
    used directly without any threshold filtering.
    
    Supports two relation formats:
      - Flat: each entry has a single 'predicate' key
      - Nested: each entry has a 'predicates' list of {predicate, predicate_score}
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"Validated relations directory does not exist: {dir_path}")
    
    scene_files = sorted(glob.glob(os.path.join(dir_path, "scene-graph_*.json")))
    if not scene_files:
        raise FileNotFoundError(
            f"No scene-graph_*.json files found in '{dir_path}'. "
            "Expected output from clean_and_refine_relations.py."
        )
    
    items = []
    for fp in scene_files:
        scene = load_json(fp)
        img_id = str(scene.get("image_id", os.path.basename(fp)))
        
        # Normalize relations: handle both flat and nested predicate formats
        raw_relations = scene.get("relations", [])
        normalized_relations = []
        
        for rel in raw_relations:
            if "predicates" in rel and isinstance(rel["predicates"], list):
                # Nested format: expand each validated predicate into its own relation entry
                for pred_entry in rel["predicates"]:
                    normalized_relations.append({
                        "subject_index": rel["subject_index"],
                        "object_index": rel["object_index"],
                        "predicate": pred_entry["predicate"],
                        "predicate_score": pred_entry.get("predicate_score", 1.0),
                        "no_relation_score": rel.get("no_relation_score", 0.0),
                    })
            elif "predicate" in rel:
                # Flat format: already one relation per entry
                normalized_relations.append({
                    "subject_index": rel["subject_index"],
                    "object_index": rel["object_index"],
                    "predicate": rel["predicate"],
                    "predicate_score": rel.get("predicate_score", 1.0),
                    "no_relation_score": rel.get("no_relation_score", 0.0),
                })
        
        scene["relations"] = normalized_relations
        items.append((img_id, scene))
    
    print(f"Loaded {len(items)} validated scenes from {dir_path}")
    total_rels = sum(len(s.get("relations", [])) for _, s in items)
    print(f"Total VLM-validated relations: {total_rels}")
    
    return items


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=False, help="Directory containing anno_merged_edges.json (threshold-based, legacy)")
    p.add_argument("--validated_relations_dir", required=False, 
                   help="Directory with VLM-validated scene-graph_*.json files from clean_and_refine_relations.py. "
                        "When provided, all validated relations become correct answers (no threshold filtering).")
    p.add_argument("--metadata_file", required=True)
    p.add_argument("--psg_meta", help="Path to custom_psg.json (optional)")
    p.add_argument("--synonyms_json", required=True)
    p.add_argument("--template_dir", required=True)
    p.add_argument("--output_questions_file", required=True)
    p.add_argument("--instances_per_template", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--generation_mode", default="required", choices=["required", "legacy", "both"])
    p.add_argument("--multiple_choice_method", action="store_true", help="Use multiple choice (A,B,C,D) for relationship questions")
    p.add_argument("--refine", action="store_true", help="Run refinement on generated questions")
    p.add_argument("--refiner_model", default="Qwen/Qwen2.5-VL-32B-Instruct", help="Model to use for refinement")
    p.add_argument("--refiner_gpus", type=int, default=4, help="Number of GPUs for refinement")
    p.add_argument("--refiner_env", help="Conda environment for refinement (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.run_dir and not args.validated_relations_dir:
        raise ValueError("Either --run_dir or --validated_relations_dir must be provided.")

    metadata = load_json(args.metadata_file)
    synonyms = load_json(args.synonyms_json)
    
    if args.validated_relations_dir:
        # NEW: Use VLM-validated relations from clean_and_refine output.
        # All relations in these files have been validated by a VLM, so
        # they are used directly as correct answers (no threshold filtering).
        print(f"Using VLM-validated relations from: {args.validated_relations_dir}")
        raw_items = load_validated_scenes(args.validated_relations_dir)
    else:
        input_path = args.run_dir
        
        # STRICT MODE: Only allow directory input containing anno_merged_edges.json
        if not os.path.isdir(input_path):
            raise ValueError(f"STRICT MODE: Input path must be a directory. Got: {input_path}")
            
        merged_edges_path = os.path.join(input_path, "anno_merged_edges.json")
        if not os.path.exists(merged_edges_path):
            raise FileNotFoundError(
                f"STRICT MODE: The directory '{input_path}' does not contain 'anno_merged_edges.json'. "
                "This script now strictly requires the merged edges output from the pipeline."
            )

        print(f"Loading merged edges strictly from: {merged_edges_path}")
        raw_items = load_merged_edges_scenes(merged_edges_path)
        
    print(f"Loaded {len(raw_items)} scenes.")

    # Collect all predicates from the loaded scenes for distractor generation
    all_predicates = set()
    for _, scene_data in raw_items:
        rels = scene_data.get("relations", [])
        for r in rels:
            if "predicate" in r:
                all_predicates.add(r["predicate"])
    all_predicates = sorted(list(all_predicates))
    print(f"Collected {len(all_predicates)} unique predicates.")

    all_questions = []

    for idx, (img_id, image_rec) in enumerate(raw_items):
        try:
            scene_struct = build_scene_struct(image_rec)
        except Exception as e:
            # We raise this to be consistent with 'strict' philosophy, or print error?
            # User said: "if the first options fails, I want to see an error and the run SHOULD fail!"
            # This referred to input loading. For scene struct building, failing globally might be too harsh 
            # if one image is bad, but let's stick to the spirit of "no hiding errors".
            print(f"Error building scene struct for {img_id}: {e}")
            raise e 

        file_name = scene_struct.get("file_name", "")
        
        if args.generation_mode in ("required", "both"):
            qs = generate_required_vqa_questions(
                scene_struct, 
                metadata, 
                synonyms, 
                all_predicates=all_predicates,
                use_multiple_choice=args.multiple_choice_method,
                scene_graph_id=img_id
            )
            for q in qs:
                all_questions.append({
                    "image_id": img_id,
                    "image_file": file_name,
                    **q,
                    "template_index": "required"
                })

    os.makedirs(os.path.dirname(args.output_questions_file), exist_ok=True)
    with open(args.output_questions_file, "w") as f:
        json.dump({"questions": all_questions}, f, indent=2)
    
    print(f"Saved {len(all_questions)} questions to {args.output_questions_file}")

    if args.refine:
        print("\n=== Running Question Refinement ===")
        refinement_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refine_questions.py")
        if not os.path.exists(refinement_script):
            print(f"Error: Refinement script not found at {refinement_script}")
            return

        cmd = [
            "python", refinement_script,
            "--input-file", args.output_questions_file,
            "--output-file", args.output_questions_file,  # In-place refinement (original is preserved in JSON field)
            "--model", args.refiner_model,
            "--num-gpus", str(args.refiner_gpus)
        ]
        
        # If env is specified, wrap in conda run
        if args.refiner_env:
            # Prepend conda run
            # cmd is currently ["python", script, ...]
            cmd = ["conda", "run", "-n", args.refiner_env, "--no-capture-output"] + cmd
            
        print(f"Executing refinement command...")
        try:
            subprocess.check_call(cmd)
            print("Refinement completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Refinement failed with exit code {e.returncode}")
            print("Note: Generated questions are still available in the output file (unrefined).")

if __name__ == "__main__":
    main()
