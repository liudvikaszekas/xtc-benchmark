# scene_struct.py
from typing import Dict, Any, List


def build_scene_struct_from_groups(image_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    PSG-style:
      groups: [{group_id, label, bbox, attrs, ...}]
      edges:  [{predicate, subject_group_id, object_group_id}]
    """
    groups = image_rec["groups"]
    edges = image_rec.get("edges", [])

    objects: List[Dict[str, Any]] = []
    group_id_to_idx: Dict[int, int] = {}

    for idx, g in enumerate(groups):
        obj = {
            "id": idx,
            "group_id": g.get("group_id", idx),
            "label": g["label"],
            "bbox": g.get("bbox"),
            "attrs": g.get("attrs", {}),
        }
        objects.append(obj)
        group_id_to_idx[obj["group_id"]] = idx

    relationships: Dict[str, Dict[int, List[int]]] = {}

    for e in edges:
        pred = e.get("predicate", e.get("best_predicate"))
        subj_gid = e["subject_group_id"]
        obj_gid = e["object_group_id"]

        if subj_gid not in group_id_to_idx or obj_gid not in group_id_to_idx:
            continue

        subj_idx = group_id_to_idx[subj_gid]
        obj_idx = group_id_to_idx[obj_gid]

        relationships.setdefault(pred, {}).setdefault(subj_idx, []).append(obj_idx)

    return {
        "image_id": image_rec.get("image_id"),
        "file_name": image_rec.get("file_name"),
        "objects": objects,
        "relationships": relationships,
    }


def build_scene_struct_from_scenegraph(image_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scene-graph style with support for member_attributes (groups):
      boxes:     [{index, label, bbox_xyxy, attributes, member_attributes?}]
      relations: [{subject_index, object_index, predicate, ...}]
    
    For groups (len(seg_ids) > 1), expands into individual member objects
    with extracted attributes from member_attributes.
    """
    objects: List[Dict[str, Any]] = []
    group_expansions: Dict[int, List[int]] = {}  # box_index -> list of member indices
    
    member_counter = 0  # Track expanded members
    
    for b in image_rec["boxes"]:
        box_index = b["index"]
        label = b["label"]
        bbox = b.get("bbox_xyxy", b.get("bbox"))
        attrs = b.get("attributes", b.get("attrs", {}))
        member_attrs = b.get("member_attributes", [])
        seg_ids = b.get("seg_ids") or []

        # Segment-only mode: only create objects for explicit segments (seg_ids).
        # If we have member_attributes, use them. Otherwise, create one member per seg_id
        # with the box-level attributes.
        if seg_ids:
            group_expansions[box_index] = []

            if member_attrs:
                for member in member_attrs:
                    seg_id = member.get("seg_id")
                    member_attr = member.get("attributes", {})
                    
                    # Filter out visual_reasoning from member attributes
                    filtered_attr = {k: v for k, v in member_attr.items() if k != "visual_reasoning"}

                    obj = {
                        "id": member_counter,
                        "original_index": box_index,
                        "box_id": b.get("id"),
                        "seg_id": seg_id,
                        "label": label,
                        "bbox": bbox,
                        "attrs": filtered_attr,
                        "is_member": True,
                    }
                    objects.append(obj)
                    group_expansions[box_index].append(member_counter)
                    member_counter += 1
            else:
                for seg_id in seg_ids:
                    # Filter out visual_reasoning from top-level attributes
                    filtered_attr = {k: v for k, v in attrs.items() if k != "visual_reasoning"} if attrs else {}
                    
                    obj = {
                        "id": member_counter,
                        "original_index": box_index,
                        "box_id": b.get("id"),
                        "seg_id": seg_id,
                        "label": label,
                        "bbox": bbox,
                        "attrs": filtered_attr,
                        "is_member": True,
                    }
                    objects.append(obj)
                    group_expansions[box_index].append(member_counter)
                    member_counter += 1

        # If there are no seg_ids, skip this box entirely (segment-only questions).

    relationships: Dict[str, Dict[int, List[int]]] = {}
    
    # Process relations - need to expand group references
    for r in image_rec.get("relations", []):
        pred = r["predicate"]
        s = r["subject_index"]
        o = r["object_index"]
        
        # Find actual object indices after expansion
        # Handle case where indices are not in group_expansions (single objects not expanded)
        # However, logic above assigns 'id' based on member_counter, not preserving original 'index' map directly.
        # We need a map from box_index -> list of new object IDs.

    # 1. First Pass: Map original box index to new object IDs
    box_idx_to_obj_ids = {}
    for obj in objects:
        orig = obj["original_index"]
        if orig not in box_idx_to_obj_ids:
            box_idx_to_obj_ids[orig] = []
        box_idx_to_obj_ids[orig].append(obj["id"])

    relationships: Dict[str, Dict[int, List[int]]] = {}
    relationship_scores: Dict[tuple, float] = {}
    
    # Process relations - need to expand group references
    for r in image_rec.get("relations", []):
        # Filter weak relations
        if r.get("no_relation_score", 0) > r.get("predicate_score", 0):
            continue
            
        pred = r["predicate"]
        score = r.get("predicate_score", 1.0)
        s = r["subject_index"]
        o = r["object_index"]
        
        # Find actual object indices
        subj_indices = box_idx_to_obj_ids.get(s, [])
        obj_indices = box_idx_to_obj_ids.get(o, [])
        
        # Create relations for all member pairs
        for si in subj_indices:
            for oi in obj_indices:
                relationships.setdefault(pred, {}).setdefault(si, []).append(oi)
                # Store score key: (subject_idx, object_idx, predicate)
                # If multiple edges map to same pair/pred (unlikely here but possible), max score?
                # For now just overwrite or keep first.
                relationship_scores[(si, oi, pred)] = score

    return {
        "image_id": image_rec.get("image_id"),
        "file_name": image_rec.get("file_name"),
        "objects": objects,
        "relationships": relationships,
        "relationship_scores": relationship_scores,
        "group_expansions": group_expansions,  # For reference
    }


def build_scene_struct(image_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-detect which format we got and convert to CLEVR-style scene_struct.
    """
    if "groups" in image_rec:
        return build_scene_struct_from_groups(image_rec)
    if "boxes" in image_rec:
        return build_scene_struct_from_scenegraph(image_rec)
    raise ValueError("Unknown scene format: expected keys 'groups' or 'boxes'.")
