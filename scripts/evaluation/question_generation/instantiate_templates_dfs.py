from collections import Counter
from itertools import combinations, product
from typing import Dict, Any, List
import random
import re

import question_engine as qeng

# Try to import VQA utilities if available
try:
    from vqa_utils import (
        QuestionNaturalizer,
        AttributeFormatter,
        ObjectDescriber,
        RelationFormatter,
        instantiate_and_clean_question
    )
    NATURALIZE_ENABLED = True
except ImportError:
    NATURALIZE_ENABLED = False


def node_shallow_copy(node):
    """Creates a copy of the given node to avoid modifying the original template node."""
    new_node = {
        "type": node["type"],
        "inputs": list(node["inputs"]),
    }
    if "side_inputs" in node:
        new_node["side_inputs"] = list(node["side_inputs"])
    return new_node


def instantiate_question_naturally(
    template: str,
    replacements: Dict[str, str],
    answer: Any = None,
    label_attr_key: str = "label"
) -> str:
    """
    Instantiate a question template naturally, hiding answer hints.
    
    Args:
        template: Template string with <key> placeholders
        replacements: Dict of placeholder -> value replacements
        answer: Answer to hide from the question
        label_attr_key: The key for label replacements
    
    Returns:
        Naturally formatted question
    """
    if not NATURALIZE_ENABLED:
        # Fallback to basic instantiation
        question = template
        for key, value in replacements.items():
            question = question.replace(f"<{key}>", str(value))
        # Remove unmatched placeholders
        question = re.sub(r"<\w+>", "", question)
        question = " ".join(question.split())
        if not question.endswith("?"):
            question += "?"
        return question.strip()
    
    try:
        # Use the VQA utilities
        question = instantiate_and_clean_question(
            template,
            replacements,
            answer
        )
        return question
    except (KeyError, ValueError) as e:
        # Fallback if natural instantiation fails
        question = template
        for key, value in replacements.items():
            question = question.replace(f"<{key}>", str(value))
        # Remove unmatched placeholders
        question = re.sub(r"<\w+>", "", question)
        question = " ".join(question.split())
        if not question.endswith("?"):
            question += "?"
        return question.strip()


def get_param_values_for_type(param_type: str, metadata, scene_struct):
    """
    Scene-aware parameter sampling.

    - Label: only labels that appear in this scene
    - Relation: only relations present in this scene
    - AttributeType: only attribute keys present in this scene (∩ metadata)
    - AttrValue: all attribute values present in this scene
    - default: fall back to metadata["types"][param_type]
    """
    objects = scene_struct["objects"]

    if param_type == "Label":
        return sorted({obj["label"] for obj in objects})

    if param_type == "Relation":
        return sorted(scene_struct.get("relationships", {}).keys())

    if param_type == "AttributeType":
        keys = set()
        for obj in objects:
            keys.update((obj.get("attrs", {}) or {}).keys())
        allowed = set(metadata["types"]["AttributeType"])
        return sorted(keys & allowed)

    if param_type == "AttrValue":
        vals = set()
        for obj in objects:
            for v in (obj.get("attrs", {}) or {}).values():
                if v in (None, "", [], {}):
                    continue
                if isinstance(v, list):
                    vals.update(v)
                else:
                    vals.add(v)
        return sorted(vals)

    if param_type == "InstanceIndex":
        return list(range(len(objects)))

    return metadata["types"].get(param_type, [])[:]


# ----------------- BERT-aware attribute equality -----------------


def attrs_equal(v1, v2) -> bool:
    """
    Compare two attribute values using your BERT-based equality
    defined in question_engine._bert_equal.

    Handles scalar or list values.
    """
    if v1 in (None, "", [], {}):
        return v2 in (None, "", [], {})
    if v2 in (None, "", [], {}):
        return False

    if not isinstance(v1, list):
        v1 = [v1]
    if not isinstance(v2, list):
        v2 = [v2]

    for a in v1:
        for b in v2:
            if qeng._bert_equal(str(a), str(b)):
                return True
    return False


def find_min_unique_attr_keys(scene_struct, obj_idx, allowed_keys=None):
    """
    For the object at obj_idx, find the smallest subset of attribute
    keys that makes it unique among all objects with the same label.

    Returns a list of attribute keys, e.g. ["color", "size"].
    Returns [] if no such subset exists or if the label is already unique.
    """
    objects = scene_struct["objects"]
    target = objects[obj_idx]
    label = target["label"]
    target_attrs = target.get("attrs", {}) or {}

    competitors = [
        o for i, o in enumerate(objects)
        if i != obj_idx and o["label"] == label
    ]
    if not competitors:
        return []  # already unique by label

    keys = list(target_attrs.keys())
    if allowed_keys is not None:
        keys = [k for k in keys if k in allowed_keys]
    if not keys:
        return []

    for r in range(1, len(keys) + 1):
        for combo in combinations(keys, r):
            unique_for_combo = True
            for comp in competitors:
                comp_attrs = comp.get("attrs", {}) or {}
                match_all = True
                for k in combo:
                    if not attrs_equal(target_attrs.get(k), comp_attrs.get(k)):
                        match_all = False
                        break
                if match_all:
                    unique_for_combo = False
                    break
            if unique_for_combo:
                return list(combo)

    return []


def find_min_unique_attr_keys_global(scene_struct, obj_idx, allowed_keys=None):
    """
    Like find_min_unique_attr_keys, but checks uniqueness
    among *all* other objects in the scene, regardless of label.
    """
    objects = scene_struct["objects"]
    target = objects[obj_idx]
    target_attrs = target.get("attrs", {}) or {}

    competitors = [o for i, o in enumerate(objects) if i != obj_idx]
    if not competitors:
        return []

    keys = list(target_attrs.keys())
    if allowed_keys is not None:
        keys = [k for k in keys if k in allowed_keys]
    if not keys:
        return []

    for r in range(1, len(keys) + 1):
        for combo in combinations(keys, r):
            unique_for_combo = True
            for comp in competitors:
                comp_attrs = comp.get("attrs", {}) or {}
                match_all = True
                for k in combo:
                    if not attrs_equal(target_attrs.get(k), comp_attrs.get(k)):
                        match_all = False
                        break
                if match_all:
                    unique_for_combo = False
                    break
            if unique_for_combo:
                return list(combo)

    return []


def build_obj_description(scene_struct, obj_idx, attr_keys, synonyms, include_label: bool = True) -> str:
    """
    Build a description from attributes.

    - include_label = True  -> 'red small chair'
    - include_label = False -> 'color red and state assembled'
    """
    obj = scene_struct["objects"][obj_idx]
    label = obj["label"]
    attrs = obj.get("attrs", {}) or {}

    # Resolve synonyms first
    resolved_attrs = {}
    for k in attr_keys:
        v = attrs.get(k)
        if v in (None, "", [], {}):
            continue
            
        # Helper to apply synonyms to a value
        def apply_synonyms(val):
            if val in synonyms:
                return random.choice(synonyms[val])
            return val

        if isinstance(v, list):
            if not v: continue
            resolved_v = [apply_synonyms(x) for x in v]
        else:
            resolved_v = apply_synonyms(v)
        
        resolved_attrs[k] = resolved_v

    if NATURALIZE_ENABLED:
        # Use ObjectDescriber
        desc_label = label if include_label else ""
        proxy_obj = {"label": desc_label, "attrs": resolved_attrs}
        desc = ObjectDescriber.describe_single_object(proxy_obj, include_attrs=True)
        return " ".join(desc.split())

    # Fallback legacy implementation
    attr_name_map = {
        "color": "color",
        "size": "size",
        "material": "material",
        "texture": "texture",
        "state": "state",
        "position": "position",
        "age": "age",
        "cleanliness": "cleanliness",
        "clothes_color": "clothes color",
        "clothes_pattern": "clothes pattern",
        "facial_expression": "facial expression",
        "gender": "gender",
        "hair_color": "hair color",
        "hair_length": "hair length",
        "pattern": "pattern",
    }

    parts = []
    # Sort keys for deterministic output in legacy mode too? 
    # Original code iterated attr_keys from argument.
    for k in attr_keys:
        if k not in resolved_attrs:
            continue
        
        v = resolved_attrs[k]
        val = v[0] if isinstance(v, list) else v

        if include_label:
            parts.append(str(val))
        else:
            pretty_name = attr_name_map.get(k, k.replace("_", " "))
            parts.append(f"{pretty_name} {val}")

    if not parts:
        return label if include_label else "object"

    if include_label:
        return " ".join(parts + [label])

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


# ----------------- DFS instantiation -----------------


def instantiate_templates_dfs(scene_struct,
                              template,
                              metadata,
                              answer_counts,
                              synonyms,
                              max_instances=None,
                              verbose=False):
    """
    DFS instantiation:
      - samples parameter values from scene-aware metadata
      - executes partial programs via qeng.answer_question
      - prunes when '__INVALID__'
    """

    label_counts = Counter(obj["label"] for obj in scene_struct["objects"])
    param_name_to_type = {p["name"]: p["type"] for p in template.get("params", [])}

    initial_state = {
        "nodes": [node_shallow_copy(template["nodes"][0])],
        "vals": {},
        "input_map": {0: 0},
        "next_template_node": 1,
    }

    states = [initial_state]
    final_states = []

    while states:
        state = states.pop()

        q = {"nodes": state["nodes"]}
        outputs = qeng.answer_question(q, metadata, scene_struct, all_outputs=True)
        answer = outputs[-1]
        if answer == "__INVALID__":
            continue

        # Constraints
        skip_state = False
        for constraint in template.get("constraints", []):
            ctype = constraint["type"]

            if ctype == "NEQ":
                p1, p2 = constraint["params"]
                v1, v2 = state["vals"].get(p1), state["vals"].get(p2)
                if v1 is not None and v2 is not None and v1 != v2:
                    skip_state = True
                    break

            elif ctype == "NULL":
                p = constraint["params"][0]
                v = state["vals"].get(p)
                if v not in (None, ""):
                    skip_state = True
                    break

            elif ctype == "OUT_NEQ":
                i, j = constraint["params"]
                i = state["input_map"].get(i)
                j = state["input_map"].get(j)
                if i is not None and j is not None and outputs[i] == outputs[j]:
                    skip_state = True
                    break

        if skip_state:
            # print("Skipping due to constraint")
            continue

        # Finished program
        if state["next_template_node"] == len(template["nodes"]):
            ans = answer
            if template.get("answer_type") == "Bool":
                ans = bool(ans)

            ans_key = ans if isinstance(ans, (str, int, float, bool)) or ans is None else str(ans)
            answer_counts[ans_key] = answer_counts.get(ans_key, 0) + 1
            
            print(f"DEBUG: Generated answer {ans}")
            state["answer"] = ans
            final_states.append(state)
            if max_instances is not None and len(final_states) >= max_instances:
                break
            continue

        # Expand next node
        tmpl_idx = state["next_template_node"]
        next_node_tmpl = template["nodes"][tmpl_idx]
        next_node = node_shallow_copy(next_node_tmpl)

        # Case A: parameter node
        if "side_inputs" in next_node and next_node["side_inputs"]:
            # Handle both single and multiple parameters
            param_names = next_node["side_inputs"]
            
            # Single parameter (original behavior)
            if len(param_names) == 1:
                param_name = param_names[0]
                param_type = param_name_to_type[param_name]

                param_values = get_param_values_for_type(param_type, metadata, scene_struct)

                # ✅ PRUNE: if selecting <ATT>, only allow attrs that exist for the chosen <OBJ_LABEL>
                if param_name == "<ATT>":
                    chosen_label = state["vals"].get("<OBJ_LABEL>")
                    chosen_idx = state["vals"].get("<OBJ_IDX>")

                    if chosen_label is not None:
                        allowed = set()
                        for obj in scene_struct["objects"]:
                            if obj["label"] != chosen_label:
                                continue
                            allowed.update((obj.get("attrs", {}) or {}).keys())
                        param_values = [a for a in param_values if a in allowed]
                    elif chosen_idx is not None:
                         if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(scene_struct["objects"]):
                             obj = scene_struct["objects"][chosen_idx]
                             allowed = set((obj.get("attrs", {}) or {}).keys())
                             param_values = [a for a in param_values if a in allowed]

                random.shuffle(param_values)

                for val in param_values:
                    # multi_instance_only on label
                    if template.get("multi_instance_only", False) and param_name == "<OBJ_LABEL>":
                        if label_counts.get(val, 0) <= 1:
                            continue

                    input_map = dict(state["input_map"])
                    input_map[tmpl_idx] = len(state["nodes"])

                    inst = {
                        "type": next_node["type"],
                        "inputs": [input_map[i] for i in next_node["inputs"]],
                        "side_inputs": [val],
                    }

                    vals = dict(state["vals"])
                    vals[param_name] = val

                    if param_name == "<ATT>" and "<ATT_TEXT>" in param_name_to_type:
                        vals["<ATT_TEXT>"] = val

                    states.append({
                        "nodes": state["nodes"] + [inst],
                        "vals": vals,
                        "input_map": input_map,
                        "next_template_node": tmpl_idx + 1,
                    })
            
            # Multiple parameters (refined templates)
            else:
                # Get values for each parameter
                param_value_lists = []
                for param_name in param_names:
                    param_type = param_name_to_type.get(param_name)
                    if param_type is None:
                        # Skip unknown params
                        param_value_lists.append([param_name])
                    else:
                        vals = get_param_values_for_type(param_type, metadata, scene_struct)
                        param_value_lists.append(vals)
                
                # Generate combinations (sample up to 5 to avoid explosion)
                all_combos = list(product(*param_value_lists))
                random.shuffle(all_combos)
                max_combos = min(10, len(all_combos))  # Limit combinations
                
                for combo in all_combos[:max_combos]:
                    input_map = dict(state["input_map"])
                    input_map[tmpl_idx] = len(state["nodes"])

                    inst = {
                        "type": next_node["type"],
                        "inputs": [input_map[i] for i in next_node["inputs"]],
                        "side_inputs": list(combo),
                    }

                    vals = dict(state["vals"])
                    for param_name, val in zip(param_names, combo):
                        vals[param_name] = val

                    states.append({
                        "nodes": state["nodes"] + [inst],
                        "vals": vals,
                        "input_map": input_map,
                        "next_template_node": tmpl_idx + 1,
                    })

        # Case B: no-parameter node
        else:
            input_map = dict(state["input_map"])
            input_map[tmpl_idx] = len(state["nodes"])

            inst = {
                "type": next_node["type"],
                "inputs": [input_map[i] for i in next_node["inputs"]],
            }

            states.append({
                "nodes": state["nodes"] + [inst],
                "vals": dict(state["vals"]),
                "input_map": input_map,
                "next_template_node": tmpl_idx + 1,
            })

    # ---------- Build output texts ----------
    text_questions, structured_questions, answers = [], [], []

    # helper: detect if template is attribute-question
    is_attr_template = any(n["type"] == "query_attr" for n in template["nodes"])

    for s in final_states:
        # Use refined templates (new format) if available, otherwise fall back to old format
        if "question_templates" in template:
            txt = random.choice(template["question_templates"])
        else:
            txt = random.choice(template.get("text", ["<ERROR: no templates>"]))
        
        asked_attr = s["vals"].get("<ATT>")

        # Prepare replacements dict for natural instantiation
        replacements = {}
        for name, val in s["vals"].items():
            rep_val = val
            if rep_val in synonyms:
                rep_val = random.choice(synonyms[rep_val])

            # avoid leaking the asked attribute via *_VAL placeholders
            if asked_attr is not None and name.startswith("<") and name.endswith(">"):
                inner = name[1:-1].lower()
                if inner.endswith("_val"):
                    base_attr = inner[:-4]
                    if base_attr == str(asked_attr).lower():
                        replacements[name[1:-1]] = ""  # Mark for removal
                        continue
            
            # Apply natural formatting to attribute types
            if NATURALIZE_ENABLED and name.startswith("<") and name.endswith(">"):
                param_name = name[1:-1]
                param_type = param_name_to_type.get(param_name)
                if param_type == "AttributeType":
                    rep_val = AttributeFormatter.format_key(rep_val)
                elif param_type == "Relation":
                    rep_val = RelationFormatter.format_relation(rep_val)

            replacements[name[1:-1]] = str(rep_val)

        # Build <OBJ_DESC> or <ANCHOR_DESC> if present
        # <OBJ_DESC> maps to the LAST unique/sample object (Target)
        # <ANCHOR_DESC> maps to the FIRST unique/sample object (Subject/Anchor)
        
        desc_placeholders = []
        if "<OBJ_DESC>" in txt: desc_placeholders.append("OBJ_DESC")
        if "<ATTRS_DESC>" in txt: desc_placeholders.append("ATTRS_DESC")
        if "<ANCHOR_DESC>" in txt: desc_placeholders.append("ANCHOR_DESC")
        if "<SUBJ_DESC>" in txt: desc_placeholders.append("SUBJ_DESC") # Alias for ANCHOR/SUBJ

        if desc_placeholders:
            outputs = qeng.answer_question({"nodes": s["nodes"]}, metadata, scene_struct, all_outputs=True)
            
            # Find all unique/sample object indices
            unique_obj_indices = []
            for i, node in enumerate(s["nodes"]):
                if node["type"] in ("sample", "unique"):
                    unique_obj_indices.append(outputs[i])
            
            if not unique_obj_indices:
                continue

            for ph in desc_placeholders:
                # Determine which object this placeholder refers to
                target_idx = None
                if ph in ("OBJ_DESC", "ATTRS_DESC"):
                    target_idx = unique_obj_indices[-1]
                elif ph in ("ANCHOR_DESC", "SUBJ_DESC"):
                    if len(unique_obj_indices) > 1:
                        target_idx = unique_obj_indices[0]
                    else:
                        # Fallback if only 1 object sampled (e.g. self-relation??)
                        target_idx = unique_obj_indices[0]
                
                if target_idx is None: 
                    continue
                    
                obj_idx = target_idx
                obj = scene_struct["objects"][obj_idx]
                obj_attrs = obj.get("attrs", {}) or {}

                # SKIP CHECK: if object has no attrs, don't generate attribute questions
                # (Only applies if we rely on attributes for the question itself, not just description)
                # But for DESCRIPTION, if we have no attributes, we just get "the chair", which is fine 
                # UNLESS uniqueness requires attributes.
                
                allowed_attr_keys = metadata["types"].get("AttributeType", [])
                
                # do not use asked attribute for disambiguation
                if asked_attr is not None:
                    allowed_attr_keys = [k for k in allowed_attr_keys if k != asked_attr]

                # Select uniqueness function and label inclusion based on placeholder
                if ph == "ATTRS_DESC":
                    # Type 2: Unique globally, NO label in description
                    attr_keys = find_min_unique_attr_keys_global(scene_struct, obj_idx, allowed_keys=allowed_attr_keys or None)
                    include_label = False
                    
                    # Must have unique attributes to be valid for Type 2
                    if not attr_keys:
                        continue
                else:
                    # Type 1/3: Unique among same label, INCLUDE label
                    attr_keys = find_min_unique_attr_keys(scene_struct, obj_idx, allowed_keys=allowed_attr_keys or None)
                    include_label = True

                    # if multiple same-label objects exist and we can't disambiguate -> skip
                    label = obj["label"]
                    competitors_same = [
                        o for j, o in enumerate(scene_struct["objects"])
                        if j != obj_idx and o["label"] == label
                    ]
                    if competitors_same and not attr_keys:
                        # Can't uniquely describe it
                        # if verbose: print(f"Cannot uniquely describe obj {obj_idx} (label {label})")
                        continue

                obj_desc = build_obj_description(scene_struct, obj_idx, attr_keys, synonyms, include_label=include_label)
                replacements[ph] = obj_desc


        # Use natural instantiation if available
        if NATURALIZE_ENABLED:
            txt = instantiate_question_naturally(
                txt,
                replacements,
                answer=s["answer"],
                label_attr_key="label"
            )
        else:
            # Fallback: old-style substitution
            for key, value in replacements.items():
                txt = txt.replace(f"<{key}>", str(value))
            # Remove any remaining placeholders
            txt = re.sub(r"<\w+>", "", txt)
            txt = " ".join(txt.split())
            if not txt.endswith("?") and not txt.endswith("."):
                txt += "?"

        txt = " ".join(txt.split())

        # Remove "-merged" and "-other" suffixes from questions and answers (noise from merged labels)
        txt = re.sub(r"-merged\b", "", txt)
        txt = re.sub(r"-other\b", "", txt)
        answer_clean = str(s["answer"]).replace("-merged", "").replace("-other", "")

        text_questions.append(txt)
        structured_questions.append(s["nodes"])
        answers.append(answer_clean)

    return text_questions, structured_questions, answers
