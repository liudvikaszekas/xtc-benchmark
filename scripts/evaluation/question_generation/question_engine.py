# question_engine.py

from __future__ import annotations
from typing import Dict, Any, List
from functools import lru_cache

from bert_score import BERTScorer

SceneStruct = Dict[str, Any]

# ---------------- BERT similarity for attribute equality ----------------

_BERT_MODEL = "roberta-large"
_BERT_DEVICE = "cpu"   # or "cpu"
_BERT_THRESH = 0.9      # tune if needed

_scorer = BERTScorer(model_type=_BERT_MODEL, device=_BERT_DEVICE)
print(f"Initialized BERTScorer with model { _BERT_MODEL } on device { _BERT_DEVICE }")


@lru_cache(maxsize=100000)
def _phrase_sim(a: str, b: str) -> float:
    """
    BERTScore F1 similarity between two short phrases.
    Cached because we'll call this a lot.
    """
    P, R, F1 = _scorer.score([a], [b])
    return float(F1[0])


def _bert_equal(a: str, b: str, thresh: float = _BERT_THRESH) -> bool:
    """
    "Equal" = BERTScore >= threshold, used INSIDE each attribute
    (color vs color, size vs size, etc.), not across attributes.
    """
    return _phrase_sim(str(a), str(b)) >= thresh


# ---------------- Core executor ----------------

def answer_question(program, metadata, scene_struct, all_outputs=False):
    objects = scene_struct["objects"]
    rels = scene_struct.get("relationships", {})
    cache = []

    for node in program["nodes"]:
        node_type = node["type"]
        inputs = [cache[i] for i in node["inputs"]]

        if node_type == "scene":
            out = list(range(len(objects)))

        elif node_type == "filter_label":
            obj_indices = inputs[0]
            label = node["side_inputs"][0]
            out = [i for i in obj_indices if objects[i]["label"] == label]

        # ✅ NEW: generic attribute filter: filter_<attr_key>
        elif node_type.startswith("filter_") and node_type != "filter_label":
            obj_indices = inputs[0]
            wanted_val = node["side_inputs"][0]
            attr_key = node_type[len("filter_"):]  # e.g. "color", "size", "pattern", ...

            out = []
            for idx in obj_indices:
                attrs = objects[idx].get("attrs", {})
                val = attrs.get(attr_key)
                if val in (None, "", [], {}):
                    continue

                # val can be scalar or list
                if not isinstance(val, list):
                    val = [val]

                ok = any(_bert_equal(str(wanted_val), str(v)) for v in val)
                if ok:
                    out.append(idx)

        elif node_type == "unique":
            obj_indices = inputs[0]
            if len(obj_indices) != 1:
                invalid = cache + ["__INVALID__"]
                return invalid if all_outputs else "__INVALID__"
            out = obj_indices[0]

        elif node_type == "sample":
            obj_indices = inputs[0]
            if not obj_indices:
                invalid = cache + ["__INVALID__"]
                return invalid if all_outputs else "__INVALID__"
            out = obj_indices[0]

        elif node_type == "relate":
            subj_idx = inputs[0]
            rel_name = node["side_inputs"][0]
            out = rels.get(rel_name, {}).get(subj_idx, [])

        elif node_type == "query_label":
            obj_idx = inputs[0]
            out = objects[obj_idx]["label"]

        elif node_type == "query_attr":
            obj_idx = inputs[0]
            attr_name = node["side_inputs"][0]
            attrs = objects[obj_idx].get("attrs", {})
            out = attrs.get(attr_name, None)

            if out in (None, "", [], {}):
                invalid = cache + ["__INVALID__"]
                return invalid if all_outputs else "__INVALID__"

            if isinstance(out, list):
                out = ", ".join(map(str, out))

        elif node_type == "exist":
            obj_indices = inputs[0]
            out = len(obj_indices) > 0

        elif node_type == "filter_index":
            obj_indices = inputs[0]
            target_idx = int(node["side_inputs"][0])
            if target_idx in obj_indices:
                out = [target_idx]
            else:
                out = []

        elif node_type == "filter_attr":
            obj_indices = inputs[0]
            attr_name = node["side_inputs"][0]   # e.g. "texture"
            wanted_val = node["side_inputs"][1]  # e.g. "woven"

            out = []
            for idx in obj_indices:
                attrs = objects[idx].get("attrs", {})
                val = attrs.get(attr_name)
                if val is None:
                    continue

                if isinstance(val, list):
                    ok = any(_bert_equal(str(wanted_val), str(v)) for v in val)
                else:
                    ok = _bert_equal(str(wanted_val), str(val))

                if ok:
                    out.append(idx)

        # --- New Nodes for Refined Templates ---

        elif node_type == "query_relation":
            subj_idx = inputs[0]
            obj_idx = inputs[1]
            found_rels = []
            
            # Search all relations
            for r, mapping in rels.items():
                if subj_idx in mapping:
                    # check if obj_idx is in the list for this subject
                    targets = mapping[subj_idx]
                    if obj_idx in targets:
                        found_rels.append(r)
            
            if not found_rels:
                 invalid = cache + ["__INVALID__"]
                 # print(f"DEBUG: No relation between {subj_idx} and {obj_idx}")
                 return invalid if all_outputs else "__INVALID__"
            
            # Return the first found relation
            # In instantiations, we might want to prioritize specific ones, 
            # but here we just take the first valid one.
            out = found_rels[0]

        elif node_type == "count":
            obj_indices = inputs[0]
            out = len(obj_indices)

        elif node_type == "compare_count":
            count1 = inputs[0]
            count2 = inputs[1]
            # Simple logic: return relation string or boolean if checking equality
            # For "Which is more common?", the answer generation logic needs to handle 
            # mapping "more" -> label1, "less" -> label2.
            # But here we just return the relation state.
            if count1 > count2:
                out = "more"
            elif count1 < count2:
                out = "less"
            else:
                out = "equal"

        elif node_type == "compare_attr":
            val1 = inputs[0]
            val2 = inputs[1]
            
            # Use BERT equality
            if isinstance(val1, list):
                val1_str = " ".join(map(str, val1))
            else:
                val1_str = str(val1)
                
            if isinstance(val2, list):
                val2_str = " ".join(map(str, val2))
            else:
                val2_str = str(val2)
                
            is_same = _bert_equal(val1_str, val2_str)
            out = "yes" if is_same else "no"

        else:
            raise ValueError(f"Unknown node type: {node_type}")

        cache.append(out)

    return cache if all_outputs else cache[-1]