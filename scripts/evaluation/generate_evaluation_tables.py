#!/usr/bin/env python3
import argparse
import json
import os
import glob
from collections import defaultdict

def compute_dataset_stats(pipeline_base: str, ige_base: str, sample_model: str = "gpt") -> dict:
    """
    Computes per-dataset statistics for Table 0:
    - Total Images
    - Total Atomic Facts (|F|)
    - Avg. Objects per Image
    - Avg. Relationships per Image
    - Avg. Attributes per Image
    - % Obj. Retr. / Attr. Query / Rel. Query

    Args:
        pipeline_base: Path to the pipeline directory containing 3_clean_and_refine_gt and 5_attributes_gt.
        ige_base: Path to the final_graphs_pt directory containing model subdirectories.
        sample_model: Model name to use for deriving question type distribution.
    """
    stats = {
        "total_images": 0,
        "total_facts": 0,
        "avg_objects": 0.0,
        "avg_relations": 0.0,
        "avg_attrs": 0.0,
        "pct_obj_retr": 0.0,
        "pct_attr_query": 0.0,
        "pct_rel_query": 0.0,
    }

    scene_graph_dir = os.path.join(pipeline_base, "3_clean_and_refine_gt")
    attr_file = os.path.join(pipeline_base, "5_attributes_gt", "attributes.jsonl")

    # --- 1. Scene Graphs: Total Images, Avg Objects, Avg Relations ---
    try:
        sg_files = glob.glob(os.path.join(scene_graph_dir, "scene-graph_*.json"))
        if not sg_files:
            print(f"Warning: No scene graph files found in {scene_graph_dir}")
        else:
            total_images = len(sg_files)
            total_objects = 0
            total_relations = 0
            for sgf in sg_files:
                try:
                    with open(sgf, "r") as f:
                        data = json.load(f)
                    total_objects += len(data.get("boxes", []))
                    total_relations += len(data.get("relations", []))
                except Exception as e:
                    print(f"Warning: Could not parse scene graph file {sgf}: {e}")
            stats["total_images"] = total_images
            stats["avg_objects"] = total_objects / total_images
            stats["avg_relations"] = total_relations / total_images
    except Exception as e:
        print(f"Warning: Could not access scene graph directory {scene_graph_dir}: {e}")

    # --- 2. Attributes: Avg Attributes per Image ---
    try:
        total_attrs = 0
        attr_obj_count = 0
        with open(attr_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                attrs = rec.get("attributes", {})
                # exclude visual_reasoning field which is not a real attribute
                total_attrs += len([k for k in attrs.keys() if k != "visual_reasoning"])
                attr_obj_count += 1
        if stats["total_images"] > 0 and attr_obj_count > 0:
            stats["avg_attrs"] = total_attrs / stats["total_images"]
        else:
            print(f"Warning: Could not compute avg attributes (images={stats['total_images']}, objs_with_attrs={attr_obj_count})")
    except FileNotFoundError:
        print(f"Warning: Attributes file not found: {attr_file}")
    except PermissionError:
        print(f"Warning: No permission to read attributes file: {attr_file}")
    except Exception as e:
        print(f"Warning: Could not process attributes file {attr_file}: {e}")

    # --- 3. Question Type Distribution: aggregate across ALL available models ---
    all_models = ["gpt", "gemini", "tar", "showo2", "showo", "bagel", "blip3o", "januspro", "mmada", "omnigen2"]
    agg_qtypes = defaultdict(int)
    total_facts = 0
    models_used = 0
    for try_model in all_models:
        try:
            ige_dir = os.path.join(ige_base, try_model)
            ige_files = glob.glob(os.path.join(ige_dir, "ige_scored_*.jsonl"))
            if not ige_files:
                continue
            ige_file = ige_files[0]
            with open(ige_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    qt = rec.get("question_type", "")
                    if "count" in qt.lower():
                        continue  # filter out all counting questions
                    score = rec.get("score")
                    if score is not None:
                        agg_qtypes[map_qtype(qt)] += 1
                        total_facts += 1
            models_used += 1
        except (PermissionError, FileNotFoundError):
            continue  # silently skip inaccessible / missing models
        except Exception as e:
            print(f"Warning: Error reading IGE file for model '{try_model}': {e}")
            continue
    if total_facts > 0:
        # Report based on a single representative model's count (divide by models_used)
        stats["total_facts"] = total_facts // models_used if models_used > 0 else total_facts
        stats["pct_obj_retr"] = agg_qtypes["Obj. Retr."] / total_facts * 100
        stats["pct_attr_query"] = agg_qtypes["Attr. Query"] / total_facts * 100
        stats["pct_rel_query"] = agg_qtypes["Rel. Query"] / total_facts * 100
    else:
        print(f"Warning: Could not load question distribution from any model in {ige_base}")

    return stats

def load_scored_jsonl(path: str) -> dict:
    records = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "question_id" in rec:
                qid_raw = rec.get("question_id")
            else:
                qid_raw = rec.get("qid")
            if qid_raw is None:
                continue
            qid = str(qid_raw)
            score = rec.get("score")
            
            # Identify object counting questions on merged objects and filter them out
            qt = rec.get("question_type", "").lower()
            qs = rec.get("question", "").lower()
            if "count" in qt and "-merged" in qs:
                continue
                
            if qid is not None and score is not None:
                try:
                    records[qid] = {
                        "score": float(score),
                        "type": rec.get("question_type", "unknown"),
                        "model_answer": str(rec.get("model_answer", "")).strip(),
                        "reference": rec.get("reference", {}),
                        "image_id": str(rec.get("image_id", ""))
                    }
                except (ValueError, TypeError):
                    pass
    return records

def map_qtype(qt: str) -> str:
    qt = qt.lower()
    if 'attribute_to_label' in qt or 'attributes_to_label' in qt or 'attribute to label' in qt or 'attributes to label' in qt:
        return 'Obj. Retr.'
    elif 'label_to_attribute' in qt or 'labels_to_attribute' in qt or 'label to attribute' in qt or 'labels to attribute' in qt:
        return 'Attr. Query'
    elif 'rel' in qt:
        return 'Rel. Query'
    elif 'count' in qt:
        return 'Counting'
    elif 'attr' in qt:
        return 'Attr. Query'
    else:
        return 'Obj. Retr.'

def is_node_matched(ref, matched_pairs):
    if not ref or not matched_pairs:
        return False
        
    ref_type = ref.get("type", "")
    
    if ref_type in ["attribute", "attribute_to_label"]:
        obj_id = str(ref.get("object_id", ""))
        return any(obj_id == str(pair[0]) or obj_id == str(pair[1]) for pair in matched_pairs)
        
    elif ref_type == "relationship":
        subj_id = str(ref.get("subject_id", ""))
        obj_id = str(ref.get("object_id", ""))
        
        subj_matched = any(subj_id == str(pair[0]) or subj_id == str(pair[1]) for pair in matched_pairs)
        obj_matched = any(obj_id == str(pair[0]) or obj_id == str(pair[1]) for pair in matched_pairs)
        
        return subj_matched and obj_matched
        
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation tables across one or more run directories."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory. Pass multiple times for multiple datasets (expects final_graphs_pt/ and vqa_outputs/).",
    )
    parser.add_argument(
        "--models",
        default="bagel,blip3o,januspro,mmada,omnigen2,showo,showo2,tar,gemini,gpt",
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--output",
        default="new_table.txt",
        help="Output table file path",
    )
    args = parser.parse_args()

    datasets = []
    for idx, run_dir in enumerate(args.run_dir, start=1):
        run_abs = os.path.abspath(run_dir)
        ds_name = f"Run {idx}"

        datasets.append(
            {
                "name": ds_name,
                "ige_base": os.path.join(run_abs, "final_graphs_pt"),
                "vqa_base": os.path.join(run_abs, "vqa_outputs"),
                "pipeline_base": run_abs,
            }
        )

    for ds in datasets:
        if not os.path.isdir(ds["ige_base"]):
            raise FileNotFoundError(f"IGE directory not found for {ds['name']}: {ds['ige_base']}")
        if not os.path.isdir(ds["vqa_base"]):
            raise FileNotFoundError(f"VQA directory not found for {ds['name']}: {ds['vqa_base']}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    
    matched_gen_stats = defaultdict(lambda: defaultdict(lambda: {"score_sum": 0.0, "count": 0}))
    all_gen_stats = defaultdict(lambda: defaultdict(lambda: {"score_sum": 0.0, "count": 0}))
    all_und_stats = defaultdict(lambda: defaultdict(lambda: {"score_sum": 0.0, "count": 0}))
    matched_und_stats = defaultdict(lambda: defaultdict(lambda: {"score_sum": 0.0, "count": 0}))
    
    # New Stats for AW-CCTA Framework
    aw_ccta_stats = defaultdict(lambda: defaultdict(lambda: {"raw_ccta_sum": 0.0, "aw_ccta_sum": 0.0, "count": 0}))
    quadrant_stats = defaultdict(lambda: {"AC": 0, "AH": 0, "GD": 0, "UD": 0, "total": 0})
    model_match_ratios = {}
    
    tau = 0.6  # Threshold for Quadrant Decomposition
    
    for model in models:
        total_gt_for_model = 0
        total_matched_for_model = 0
        
        for dataset in datasets:
            ige_base = dataset["ige_base"]
            vqa_base = dataset["vqa_base"]

            match_file = os.path.join(ige_base, model, "scene_graphs_matched.json")
            try:
                with open(match_file, "r") as f:
                    matched_data = json.load(f)
                per_image_results = matched_data.get("per_image_results", {})
                
                matched_nodes_lookup = {
                    str(img_id): data.get("matched_node_pairs", [])
                    for img_id, data in per_image_results.items()
                }
                
                total_matched = sum(data.get("num_matched_nodes", 0) for data in per_image_results.values())
                total_gt = sum(data.get("num_gt_nodes", 0) for data in per_image_results.values())
                
                total_matched_for_model += total_matched
                total_gt_for_model += total_gt
                
            except Exception as e:
                print(f"Warning: Error loading {match_file} for model {model}: {e}")
                continue

            ige_dir = os.path.join(ige_base, model)
            ige_files = glob.glob(os.path.join(ige_dir, "ige_scored_*.jsonl"))
            if not ige_files:
                print(f"Warning: No IGE scored files found for model {model} in {ige_dir}")
                continue
            ige_file = ige_files[0]
            
            try:
                ige_scores = load_scored_jsonl(ige_file)
            except Exception as e:
                print(f"Warning: Error loading IGE scores for model {model} from {ige_file}: {e}")
                continue
                
            vqa_dir = os.path.join(vqa_base, model)
            vqa_files = glob.glob(os.path.join(vqa_dir, "scored_*_fix.jsonl"))
            if not vqa_files:
                vqa_files = glob.glob(os.path.join(vqa_dir, "scored_*.jsonl"))
            if not vqa_files:
                print(f"Warning: No VQA scored files found for model {model} in {vqa_dir}")
                continue
            vqa_file = vqa_files[-1]
            
            try:
                vqa_scores = load_scored_jsonl(vqa_file)
            except Exception as e:
                print(f"Warning: Error loading VQA scores for model {model} from {vqa_file}: {e}")
                continue
                
            if not ige_scores or not vqa_scores:
                print(f"Warning: Empty scores for model {model} in {vqa_base} or {ige_base}")
                continue
                
            common_ids = set(ige_scores.keys()) & set(vqa_scores.keys())
            if not common_ids:
                print(f"Warning: No common IDs found between IGE and VQA for model {model} in {vqa_base} or {ige_base}")
                continue
                
            for qid in common_ids:
                g_item = ige_scores[qid]
                u_item = vqa_scores[qid]
                
                img_id = g_item["image_id"]
                ref = g_item["reference"]
                q_type = map_qtype(g_item["type"])
                
                g_raw = g_item["score"]
                u_f = u_item["score"]
                
                all_und_stats[model][q_type]["score_sum"] += u_f
                all_und_stats[model][q_type]["count"] += 1
                
                all_gen_stats[model][q_type]["score_sum"] += g_raw
                all_gen_stats[model][q_type]["count"] += 1
                
                # Record overall generation scores for all nodes
                all_gen_stats[model]["Overall"]["score_sum"] += g_raw
                all_gen_stats[model]["Overall"]["count"] += 1
                
                matched_pairs_for_img = matched_nodes_lookup.get(img_id, [])
                is_matched = is_node_matched(ref, matched_pairs_for_img)
                
                if is_matched:
                    matched_gen_stats[model][q_type]["score_sum"] += g_raw
                    matched_gen_stats[model][q_type]["count"] += 1
                    
                    # Record overall generation scores for matched nodes only
                    matched_gen_stats[model]["Overall"]["score_sum"] += g_raw
                    matched_gen_stats[model]["Overall"]["count"] += 1
                    
                    matched_und_stats[model][q_type]["score_sum"] += u_f
                    matched_und_stats[model][q_type]["count"] += 1
                    
                # --- AW-CCTA AND QUADRANT LOGIC ---
                # If unmatched, force generation score to 0.0 for alignment metrics
                g_f_eff = g_raw if is_matched else 0.0
                
                raw_ccta = 1.0 - abs(g_f_eff - u_f)
                aw_ccta = raw_ccta * ((g_f_eff + u_f) / 2.0)
                
                # Record Global CCTA/AW-CCTA
                aw_ccta_stats[model][q_type]["raw_ccta_sum"] += raw_ccta
                aw_ccta_stats[model][q_type]["aw_ccta_sum"] += aw_ccta
                aw_ccta_stats[model][q_type]["count"] += 1
                
                aw_ccta_stats[model]["Overall"]["raw_ccta_sum"] += raw_ccta
                aw_ccta_stats[model]["Overall"]["aw_ccta_sum"] += aw_ccta
                aw_ccta_stats[model]["Overall"]["count"] += 1
                
                # Quadrant Assignment
                if g_f_eff >= tau and u_f >= tau:
                    quadrant_stats[model]["AC"] += 1
                elif g_f_eff < tau and u_f < tau:
                    quadrant_stats[model]["AH"] += 1
                elif g_f_eff >= tau and u_f < tau:
                    quadrant_stats[model]["GD"] += 1
                elif g_f_eff < tau and u_f >= tau:
                    quadrant_stats[model]["UD"] += 1
                    
                quadrant_stats[model]["total"] += 1
                
        model_match_ratios[model] = (total_matched_for_model / total_gt_for_model) * 100 if total_gt_for_model > 0 else 0.0
                
    if not model_match_ratios:
        print("No models processed successfully. Please ensure you have access to the matched JSON data.")
        return
        
    # --- Compute Dataset Statistics for Table 0 ---
    print("Computing dataset statistics for Table 0...")
    dataset_stats = {}
    for dataset in datasets:
        ds_name = dataset["name"]
        pipeline_base = dataset["pipeline_base"]
        ige_base = dataset["ige_base"]
        print(f"  Processing {ds_name}...")
        dataset_stats[ds_name] = compute_dataset_stats(pipeline_base, ige_base)

    output_path = args.output
    with open(output_path, "w") as f:

        # --- TABLE 0: Dataset Statistics ---
        f.write("Table 0: Dataset Statistics\n")
        f.write("| Dataset | Total Images | Total Facts (|F|) | Avg. Obj/Img | Avg. Rel/Img | Avg. Attr/Img | % Obj. Retr. | % Attr. Query | % Rel. Query |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for dataset in datasets:
            ds_name = dataset["name"]
            s = dataset_stats.get(ds_name, {})
            ti = s.get("total_images", 0)
            tf = s.get("total_facts", 0)
            ao = s.get("avg_objects", 0.0)
            ar = s.get("avg_relations", 0.0)
            aa = s.get("avg_attrs", 0.0)
            po = s.get("pct_obj_retr", 0.0)
            pa = s.get("pct_attr_query", 0.0)
            pr = s.get("pct_rel_query", 0.0)
            ti_str = str(ti) if ti > 0 else "N/A"
            tf_str = str(tf) if tf > 0 else "N/A"
            ao_str = f"{ao:.2f}" if ao > 0 else "N/A"
            ar_str = f"{ar:.2f}" if ar > 0 else "N/A"
            aa_str = f"{aa:.2f}" if aa > 0 else "N/A"
            po_str = f"{po:.1f}%" if po > 0 else "N/A"
            pa_str = f"{pa:.1f}%" if pa > 0 else "N/A"
            pr_str = f"{pr:.1f}%" if pr > 0 else "N/A"
            f.write(f"| {ds_name} | {ti_str} | {tf_str} | {ao_str} | {ar_str} | {aa_str} | {po_str} | {pa_str} | {pr_str} |\n")
        f.write("\n")
        
        # --- TABLE 1 UPDATED ---
        f.write("Table 1: Matched Nodes and Generation Scores\n")
        f.write("| Model | % Matched Nodes | Overall Gen. (All) | Overall Gen. (Matched) | Attr. Score (All) | Rel. Score (All) | Attr. Score (Matched) | Rel. Score (Matched) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for model in sorted(model_match_ratios.keys()):
            pct = model_match_ratios[model]
            
            st_g_all = all_gen_stats[model]
            gen_overall_all = st_g_all["Overall"]["score_sum"] / max(st_g_all["Overall"]["count"], 1) if st_g_all["Overall"]["count"] > 0 else 0.0
            attr_all = st_g_all["Attr. Query"]["score_sum"] / max(st_g_all["Attr. Query"]["count"], 1) if st_g_all["Attr. Query"]["count"] > 0 else 0.0
            rel_all = st_g_all["Rel. Query"]["score_sum"] / max(st_g_all["Rel. Query"]["count"], 1) if st_g_all["Rel. Query"]["count"] > 0 else 0.0
            
            st_g = matched_gen_stats[model]
            gen_overall_mat = st_g["Overall"]["score_sum"] / max(st_g["Overall"]["count"], 1) if st_g["Overall"]["count"] > 0 else 0.0
            attr_g = st_g["Attr. Query"]["score_sum"] / max(st_g["Attr. Query"]["count"], 1) if st_g["Attr. Query"]["count"] > 0 else 0.0
            rel_g = st_g["Rel. Query"]["score_sum"] / max(st_g["Rel. Query"]["count"], 1) if st_g["Rel. Query"]["count"] > 0 else 0.0
            
            f.write(f"| {model} | {pct:.1f}% | {gen_overall_all:.3f} | {gen_overall_mat:.3f} | {attr_all:.3f} | {rel_all:.3f} | {attr_g:.3f} | {rel_g:.3f} |\n")
            
        f.write("\n")
        
        f.write("Table 2: Understanding Scores\n")
        f.write("| Model | Overall Und. | Obj. Retr. | Rel. Query | Attr. Query | Matched Rel. | Matched Attr. |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for model in sorted(model_match_ratios.keys()):
            st_u_all = all_und_stats[model]
            st_u_mat = matched_und_stats[model]
            
            # Calculate overall based on Obj. Retr., Rel. Query, and Attr. Query only
            sum_scores = st_u_all["Obj. Retr."]["score_sum"] + st_u_all["Rel. Query"]["score_sum"] + st_u_all["Attr. Query"]["score_sum"]
            sum_counts = st_u_all["Obj. Retr."]["count"] + st_u_all["Rel. Query"]["count"] + st_u_all["Attr. Query"]["count"]
            overall = sum_scores / sum_counts if sum_counts > 0 else 0.0
            
            obj = st_u_all["Obj. Retr."]["score_sum"] / max(st_u_all["Obj. Retr."]["count"], 1) if st_u_all["Obj. Retr."]["count"] > 0 else 0.0
            rel = st_u_all["Rel. Query"]["score_sum"] / max(st_u_all["Rel. Query"]["count"], 1) if st_u_all["Rel. Query"]["count"] > 0 else 0.0
            attr = st_u_all["Attr. Query"]["score_sum"] / max(st_u_all["Attr. Query"]["count"], 1) if st_u_all["Attr. Query"]["count"] > 0 else 0.0
            
            mat_rel = st_u_mat["Rel. Query"]["score_sum"] / max(st_u_mat["Rel. Query"]["count"], 1) if st_u_mat["Rel. Query"]["count"] > 0 else 0.0
            mat_attr = st_u_mat["Attr. Query"]["score_sum"] / max(st_u_mat["Attr. Query"]["count"], 1) if st_u_mat["Attr. Query"]["count"] > 0 else 0.0
            
            f.write(f"| {model} | {overall:.3f} | {obj:.3f} | {rel:.3f} | {attr:.3f} | {mat_rel:.3f} | {mat_attr:.3f} |\n")
            
        f.write("\n")
        
        f.write("Table 3: Global Alignment Scores (Raw CCTA)\n")
        f.write("| Model | Overall | Attr. Query | Rel. Query |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        for model in sorted(model_match_ratios.keys()):
            st_ccta = aw_ccta_stats[model]
            
            raw_overall = st_ccta["Overall"]["raw_ccta_sum"] / max(st_ccta["Overall"]["count"], 1)
            raw_attr = st_ccta["Attr. Query"]["raw_ccta_sum"] / max(st_ccta["Attr. Query"]["count"], 1)
            raw_rel = st_ccta["Rel. Query"]["raw_ccta_sum"] / max(st_ccta["Rel. Query"]["count"], 1)
            
            f.write(f"| {model} | {raw_overall:.3f} | {raw_attr:.3f} | {raw_rel:.3f} |\n")

        f.write("\n")
        
        f.write("Table 4: Global Alignment Scores (AW-CCTA)\n")
        f.write("| Model | Overall | Attr. Query | Rel. Query |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        for model in sorted(model_match_ratios.keys()):
            st_ccta = aw_ccta_stats[model]
            
            aw_overall = st_ccta["Overall"]["aw_ccta_sum"] / max(st_ccta["Overall"]["count"], 1)
            aw_attr = st_ccta["Attr. Query"]["aw_ccta_sum"] / max(st_ccta["Attr. Query"]["count"], 1)
            aw_rel = st_ccta["Rel. Query"]["aw_ccta_sum"] / max(st_ccta["Rel. Query"]["count"], 1)
            
            f.write(f"| {model} | {aw_overall:.3f} | {aw_attr:.3f} | {aw_rel:.3f} |\n")

        f.write("\n")
        
        f.write(f"Table 5: Quadrant Distributions (tau={tau})\n")
        f.write("| Model | Aligned Comp. (AC) | Aligned Halluc. (AH) | Gen. Dom. (GD) | Und. Dom. (UD) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        for model in sorted(model_match_ratios.keys()):
            q_stats = quadrant_stats[model]
            total = max(q_stats["total"], 1)
            
            p_ac = (q_stats["AC"] / total) * 100
            p_ah = (q_stats["AH"] / total) * 100
            p_gd = (q_stats["GD"] / total) * 100
            p_ud = (q_stats["UD"] / total) * 100
            
            f.write(f"| {model} | {p_ac:.1f}% | {p_ah:.1f}% | {p_gd:.1f}% | {p_ud:.1f}% |\n")

    print(f"Output successfully written to {output_path}")

if __name__ == "__main__":
    main()
