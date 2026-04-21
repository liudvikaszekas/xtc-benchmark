#!/usr/bin/env python3
"""
Consistency metrics between VQA and IGE scores.

Computes:
    - raw consistency agreement (CCTA/CA): E_f[1 - |v_f - u_f|]
    - quadrant decomposition with threshold tau
    - consistency integrity ratio (CIR)
    - accuracy-weighted consistency (AW-CCTA):
            E_f[(1 - |v_f-u_f|) * (v_f+u_f)/2]
    - decomposition terms for AW-CCTA

Inputs:
    --vqa-scored    : scored JSONL from VQA judging (has "question_id", "score")
    --ige-scored    : scored JSONL from IGE judging (has "question_id", "score")
    --threshold     : quadrant threshold tau (default: 0.6)

Output:
    --output        : JSON with global/per-type metrics and per-fact records.
"""

import argparse
import json
import os
from collections import defaultdict

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
            "consistency_agreement": 0.0,
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
        "consistency_agreement": round(ccta, 6),
        "ccta": round(ccta, 6),
        "aw_ccta": round(aw_ccta, 6),
        "mean_joint_accuracy": round(mean_joint_accuracy, 6),
        "alignment_accuracy_covariance": round(covariance, 6),
        "quadrant_counts": quadrant_counts,
        "quadrant_proportions": quadrant_props,
        "consistency_integrity_ratio": round(cir, 6),
        "threshold": threshold,
    }


def load_scored_jsonl(path: str) -> dict:
    """Return {question_id -> record} from a scored JSONL file."""
    records = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("question_id") or rec.get("qid")
            if qid is not None:
                records[qid] = rec
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Compute Consistency Agreement between VQA and IGE scored results."
    )
    parser.add_argument("--vqa-scored", required=True,
                        help="Path to scored JSONL from VQA LLM judge.")
    parser.add_argument("--ige-scored", required=True,
                        help="Path to scored JSONL from IGE LLM judge.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path for consistency agreement results.")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Quadrant threshold tau for AC/AH/GD/UD assignment (default: 0.6).")
    args = parser.parse_args()

    # Load both scored sets
    print(f"Loading VQA scores from {args.vqa_scored}")
    vqa = load_scored_jsonl(args.vqa_scored)
    print(f"  Loaded {len(vqa)} VQA scored records.")

    print(f"Loading IGE scores from {args.ige_scored}")
    ige = load_scored_jsonl(args.ige_scored)
    print(f"  Loaded {len(ige)} IGE scored records.")

    # Find common question IDs with valid scores in both
    common_ids = set(vqa.keys()) & set(ige.keys())
    print(f"  Common question IDs: {len(common_ids)}")

    # Filter to questions with valid (non-None) scores on both sides
    facts = []
    for qid in sorted(common_ids, key=lambda x: int(x) if str(x).isdigit() else x):
        vqa_score = vqa[qid].get("score")   # normalised [0, 1]
        ige_score = ige[qid].get("score")

        if vqa_score is None or ige_score is None:
            continue

        v = clamp01(vqa_score)
        u = clamp01(ige_score)
        agreement = 1.0 - abs(v - u)
        joint_accuracy = (v + u) / 2.0
        aw_term = agreement * joint_accuracy
        quadrant = assign_quadrant(v, u, args.threshold)

        facts.append({
            "question_id": qid,
            "question_type": vqa[qid].get("question_type") or ige[qid].get("question_type"),
            "question": vqa[qid].get("question", ""),
            "gt_answer": vqa[qid].get("gt_answer", ""),
            "vqa_model_answer": vqa[qid].get("model_answer", ""),
            "ige_model_answer": ige[qid].get("model_answer", ""),
            "vqa_score": v,
            "ige_score": u,
            "agreement": round(agreement, 6),
            "joint_accuracy": round(joint_accuracy, 6),
            "aw_term": round(aw_term, 6),
            "quadrant": quadrant,
        })

    if not facts:
        print("WARNING: No overlapping facts with valid scores in both sets!")

    overall = summarize_facts(facts, args.threshold)

    by_type_facts = defaultdict(list)
    for f in facts:
        qt = f.get("question_type") or "unknown"
        by_type_facts[qt].append(f)

    type_breakdown = {}
    for qt in sorted(by_type_facts.keys()):
        type_breakdown[qt] = summarize_facts(by_type_facts[qt], args.threshold)

    result = {
        "consistency_agreement": overall["consistency_agreement"],
        "ccta": overall["ccta"],
        "aw_ccta": overall["aw_ccta"],
        "mean_joint_accuracy": overall["mean_joint_accuracy"],
        "alignment_accuracy_covariance": overall["alignment_accuracy_covariance"],
        "consistency_integrity_ratio": overall["consistency_integrity_ratio"],
        "quadrant_counts": overall["quadrant_counts"],
        "quadrant_proportions": overall["quadrant_proportions"],
        "threshold": args.threshold,
        "num_facts": overall["num_facts"],
        "num_vqa_scored": len(vqa),
        "num_ige_scored": len(ige),
        "num_common_ids": len(common_ids),
        "by_question_type": type_breakdown,
        "per_fact": facts,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Consistency Agreement (CCTA): {overall['ccta']:.4f}")
    print(f"AW-CCTA:                      {overall['aw_ccta']:.4f}")
    print(f"CIR:                          {overall['consistency_integrity_ratio']:.4f}")
    print(f"Number of facts:              {overall['num_facts']}")
    print(f"{'='*60}")
    if type_breakdown:
        print("\nBreakdown by question type:")
        for qt, v in type_breakdown.items():
            print(
                f"  {qt}: "
                f"CCTA={v['ccta']:.4f}  AW-CCTA={v['aw_ccta']:.4f}  "
                f"CIR={v['consistency_integrity_ratio']:.4f}  (n={v['num_facts']})"
            )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
