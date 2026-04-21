#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


QUADRANTS = ("AC", "AH", "GD", "UD")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def assign_quadrant(vqa_score: float, ige_score: float, threshold: float) -> str:
    if vqa_score >= threshold and ige_score >= threshold:
        return "AC"
    if vqa_score < threshold and ige_score < threshold:
        return "AH"
    if vqa_score >= threshold and ige_score < threshold:
        return "GD"
    return "UD"


def summarize_facts(facts: list, threshold: float) -> dict:
    if not facts:
        return {
            "num_facts": 0,
            "ccta": 0.0,
            "aw_ccta": 0.0,
            "mean_joint_accuracy": 0.0,
            "alignment_accuracy_covariance": 0.0,
            "quadrant_counts": {q: 0 for q in QUADRANTS},
            "quadrant_proportions": {q: 0.0 for q in QUADRANTS},
            "consistency_integrity_ratio": 0.0,
            "threshold": threshold,
        }

    n = len(facts)
    agreements = [f["agreement"] for f in facts]
    joint_acc = [f["joint_accuracy"] for f in facts]
    aw_terms = [f["aw_term"] for f in facts]

    ccta = sum(agreements) / n
    mean_joint_accuracy = sum(joint_acc) / n
    aw_ccta = sum(aw_terms) / n
    covariance = aw_ccta - (ccta * mean_joint_accuracy)

    quadrant_counts = {q: 0 for q in QUADRANTS}
    for f in facts:
        quadrant_counts[f["quadrant"]] += 1
    quadrant_props = {q: round(quadrant_counts[q] / n, 6) for q in QUADRANTS}

    ac = quadrant_props["AC"]
    ah = quadrant_props["AH"]
    cir = ac / (ac + ah) if (ac + ah) > 0 else 0.0

    return {
        "num_facts": n,
        "ccta": round(ccta, 6),
        "aw_ccta": round(aw_ccta, 6),
        "mean_joint_accuracy": round(mean_joint_accuracy, 6),
        "alignment_accuracy_covariance": round(covariance, 6),
        "quadrant_counts": quadrant_counts,
        "quadrant_proportions": quadrant_props,
        "consistency_integrity_ratio": round(cir, 6),
        "threshold": threshold,
    }


def load_scored_jsonl(path: Path) -> dict:
    records = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("question_id") or rec.get("qid")
            if qid is None:
                continue
            records[str(qid)] = rec
    return records


def is_node_matched(reference: dict, matched_pairs: list) -> bool:
    if not reference or not matched_pairs:
        return False

    ref_type = str(reference.get("type", ""))

    if ref_type in {"attribute", "attribute_to_label"}:
        object_id = str(reference.get("object_id", ""))
        return any(object_id == str(pair[0]) or object_id == str(pair[1]) for pair in matched_pairs)

    if ref_type == "relationship":
        subject_id = str(reference.get("subject_id", ""))
        object_id = str(reference.get("object_id", ""))
        subject_matched = any(subject_id == str(pair[0]) or subject_id == str(pair[1]) for pair in matched_pairs)
        object_matched = any(object_id == str(pair[0]) or object_id == str(pair[1]) for pair in matched_pairs)
        return subject_matched and object_matched

    return False


def build_fact(vqa_rec: dict, ige_rec: dict, threshold: float) -> dict | None:
    vqa_score = vqa_rec.get("score")
    ige_score = ige_rec.get("score")
    if vqa_score is None or ige_score is None:
        return None

    v = clamp01(vqa_score)
    u = clamp01(ige_score)
    agreement = 1.0 - abs(v - u)
    joint_accuracy = (v + u) / 2.0
    aw_term = agreement * joint_accuracy
    quadrant = assign_quadrant(v, u, threshold)

    return {
        "agreement": agreement,
        "joint_accuracy": joint_accuracy,
        "aw_term": aw_term,
        "quadrant": quadrant,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute AW-CCTA on all-overlap and matched-only facts for a run directory."
    )
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory (expects final_graphs_pt/<model>)")
    parser.add_argument("--vqa-dir", required=True, help="Directory containing per-model scored_<model>.jsonl")
    parser.add_argument("--threshold", type=float, default=0.6, help="Quadrant threshold tau (default: 0.6)")
    parser.add_argument("--suffix", default="", help="Optional IGE suffix, e.g. '_equal'")
    parser.add_argument("--output", default=None, help="Optional summary output JSON path")
    args = parser.parse_args()

    final_graphs = Path(args.run_dir) / "final_graphs_pt"
    vqa_dir = Path(args.vqa_dir)

    models = sorted([p.name for p in final_graphs.iterdir() if p.is_dir()])
    if not models:
        print(f"No model directories found in {final_graphs}")
        return

    summary = {
        "run_dir": str(args.run_dir),
        "vqa_dir": str(args.vqa_dir),
        "threshold": args.threshold,
        "suffix": args.suffix,
        "models": {},
    }

    for model in models:
        vqa_scored = vqa_dir / model / f"scored_{model}.jsonl"
        ige_scored = final_graphs / model / f"ige_scored_{model}{args.suffix}.jsonl"
        matched_graphs = final_graphs / model / "scene_graphs_matched.json"

        if not (vqa_scored.exists() and ige_scored.exists() and matched_graphs.exists()):
            print(f"[skip] {model} (missing scored or matched file)")
            continue

        vqa = load_scored_jsonl(vqa_scored)
        ige = load_scored_jsonl(ige_scored)
        matched = json.loads(matched_graphs.read_text())
        per_image = matched.get("per_image_results", {})

        common_ids = set(vqa.keys()) & set(ige.keys())
        all_facts = []
        matched_facts = []

        for qid in sorted(common_ids, key=lambda x: int(x) if str(x).isdigit() else x):
            vqa_rec = vqa[qid]
            ige_rec = ige[qid]
            fact = build_fact(vqa_rec, ige_rec, args.threshold)
            if fact is None:
                continue

            all_facts.append(fact)

            image_id = str(ige_rec.get("image_id", ""))
            reference = ige_rec.get("reference") or {}
            matched_pairs = (per_image.get(image_id) or {}).get("matched_node_pairs", [])
            if is_node_matched(reference, matched_pairs):
                matched_facts.append(fact)

        overall_metrics = summarize_facts(all_facts, args.threshold)
        matched_metrics = summarize_facts(matched_facts, args.threshold)

        result = {
            "model": model,
            "num_vqa_scored": len(vqa),
            "num_ige_scored": len(ige),
            "num_common_ids": len(common_ids),
            "num_all_facts": len(all_facts),
            "num_matched_facts": len(matched_facts),
            "all_overlap": overall_metrics,
            "matched_only": matched_metrics,
        }

        out_file = final_graphs / model / f"aw_ccta_comparison_{model}{args.suffix}.json"
        out_file.write_text(json.dumps(result, indent=2))
        summary["models"][model] = result

        print(
            f"[done] {model}: all AW-CCTA={overall_metrics['aw_ccta']:.6f} "
            f"(n={overall_metrics['num_facts']}), matched AW-CCTA={matched_metrics['aw_ccta']:.6f} "
            f"(n={matched_metrics['num_facts']})"
        )

    output_path = Path(args.output) if args.output else (Path(args.run_dir) / "aw_ccta_comparison_summary.json")
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()