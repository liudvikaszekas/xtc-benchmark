#!/usr/bin/env python3
"""Run forced None-recovery experiments for IGE answers.

Workflow per model:
1. Sample question IDs where IGE answer is "None." for:
   - label_to_attribute
   - label_attributes_to_relationship
2. Generate replacement answers for sampled attribute rows using Qwen2-VL-72B
   on model-generated images under coco_images/<model>/.
3. For sampled relationship rows, insert GT answer if:
   - subject/object are matched in IGE debug fields, and
   - GT relation appears in top-3 relation candidates from <model>-scene-graph.pkl.
4. Judge augmented IGE rows with existing llm_judge.py.
5. Compute baseline and augmented CCTA/AW-CCTA with consistency_agreement.py
   restricted to:
   - question types label_to_attribute, label_attributes_to_relationship
   - non-"None." answers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import re
import subprocess
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams


TARGET_TYPES = {"label_to_attribute", "label_attributes_to_relationship"}
ATTR_TYPE = "label_to_attribute"
REL_TYPE = "label_attributes_to_relationship"


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Forced None-recovery experiment")
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--models", nargs="+", default=["gemini", "blip3o", "mmada"])
    parser.add_argument("--samples-per-type", type=int, default=100)
    parser.add_argument("--default-set-samples", type=int, default=200)
    parser.add_argument("--forced-set-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="Qwen/Qwen2-VL-72B-Instruct")
    parser.add_argument("--attribute-model", default=None,
                        help="Model for attribute generation (defaults to --judge-model).")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-gen-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-attribute-generation", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def is_none_answer(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip().lower()
    return s in {"none", "none."}


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def split_gt_relation_candidates(gt_answer: str) -> List[str]:
    # Keep this permissive to handle composite GT relations.
    parts = re.split(r"\s*(?:,| and | or )\s*", gt_answer.strip(), flags=re.IGNORECASE)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [gt_answer.strip()]


def sample_none_rows(
    rows: List[Dict[str, Any]],
    qtype: str,
    sample_n: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], int]:
    candidates = [r for r in rows if r.get("question_type") == qtype and is_none_answer(r.get("model_answer"))]
    if len(candidates) <= sample_n:
        return candidates, max(0, sample_n - len(candidates))
    return rng.sample(candidates, sample_n), 0


def build_qwen_prompt(question: str) -> str:
    placeholder = "<|image_pad|>"
    return (
        "<|im_start|>system\n"
        "You are a concise visual question answering assistant. "
        "Answer with a short phrase only, no explanation.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|>{placeholder}<|vision_end|>{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def init_qwen_llm(model_name: str, num_gpus: int, gpu_mem_util: float, max_model_len: int) -> LLM:
    return LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )


def destroy_llm(llm: Optional[LLM]) -> None:
    """Delete LLM instance and release GPU VRAM so subprocess judges can load their own copy."""
    if llm is None:
        return
    import torch
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[cleanup] LLM destroyed and GPU VRAM freed.", flush=True)


def generate_attribute_answers(
    llm: LLM,
    model_name: str,
    attr_rows: List[Dict[str, Any]],
    image_dir: Path,
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    inputs = []
    meta: List[Dict[str, Any]] = []

    path_cache: Dict[str, Path | None] = {}

    def resolve_image_path(image_id: str, image_name: str) -> Path | None:
        key = f"{image_id}::{image_name}"
        if key in path_cache:
            return path_cache[key]

        names = [f"{image_id}.jpg", image_name]
        if str(image_id).isdigit():
            names.append(f"{int(image_id):012d}.jpg")

        # Fast checks in the given folder first.
        for name in names:
            p = image_dir / name
            if p.exists():
                path_cache[key] = p
                return p

        # Fallback for nested layouts: search recursively by filename.
        for name in names:
            hits = list(image_dir.glob(f"**/{name}"))
            if hits:
                path_cache[key] = hits[0]
                return hits[0]

        path_cache[key] = None
        return None

    for row in attr_rows:
        qid = int(row["question_id"])
        image_id = str(row["image_id"])
        img_name = row.get("image_name", f"{image_id}.jpg")
        resolved = resolve_image_path(image_id, img_name)

        if resolved is None:
            meta.append(
                {
                    "question_id": qid,
                    "image_id": image_id,
                    "status": "missing_image",
                    "image_path": str(image_dir / img_name),
                }
            )
            continue

        image = Image.open(resolved).convert("RGB")
        inputs.append(
            {
                "prompt": build_qwen_prompt(row["question"]),
                "multi_modal_data": {"image": image},
                "multi_modal_uuids": {"image": str(resolved)},
            }
        )
        meta.append(
            {
                "question_id": qid,
                "image_id": image_id,
                "status": "queued",
                "image_path": str(resolved),
                "question": row["question"],
            }
        )

    answers: Dict[int, str] = {}
    if not inputs:
        return answers, meta

    outputs = llm.generate(inputs, sampling)
    out_idx = 0
    for m in meta:
        if m["status"] != "queued":
            continue
        text = outputs[out_idx].outputs[0].text.strip()
        out_idx += 1
        if not text:
            m["status"] = "empty_output"
            continue
        m["status"] = "ok"
        m["generated_answer"] = text
        answers[int(m["question_id"])] = text

    return answers, meta


def load_predicate_classes(root: Path) -> List[str]:
    p = root / "psg_generation" / "output_kmax" / "custom_psg.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    preds = data.get("predicate_classes", [])
    if not preds:
        raise RuntimeError(f"No predicate_classes in {p}")
    return preds


def build_seg_to_box_index(pred_sg_path: Path) -> Dict[str, int]:
    data = json.loads(pred_sg_path.read_text(encoding="utf-8"))
    mapping: Dict[str, int] = {}
    for box_idx, box in enumerate(data.get("boxes", [])):
        idx = int(box.get("index", box_idx))
        mapping[str(box.get("id"))] = idx
        for sid in box.get("seg_ids", []) or []:
            mapping[str(sid)] = idx
        for member in box.get("member_attributes", []) or []:
            sid = member.get("seg_id")
            if sid is not None:
                mapping[str(sid)] = idx
    return mapping


def top3_predicates_for_pair(
    scene_entry: Dict[str, Any],
    subj_idx: int,
    obj_idx: int,
    predicate_classes: List[str],
) -> List[str]:
    pairs = scene_entry["pairs"]
    rel_scores = scene_entry["rel_scores"]

    hit = np.where((pairs[:, 0] == subj_idx) & (pairs[:, 1] == obj_idx))[0]
    if hit.size == 0:
        return []

    score_vec = rel_scores[int(hit[0])]
    # score_vec index 0 is implicit "no relation"; predicate_classes are 1..N.
    candidate_indices = list(range(1, int(score_vec.shape[0])))
    candidate_indices.sort(key=lambda i: float(score_vec[i]), reverse=True)
    top3_idx = candidate_indices[:3]
    return [predicate_classes[i - 1] for i in top3_idx]


def build_augmented_rows(
    model: str,
    root: Path,
    base_rows: List[Dict[str, Any]],
    sampled_attr: List[Dict[str, Any]],
    sampled_rel: List[Dict[str, Any]],
    generated_attr_answers: Dict[int, str],
    predicate_classes: List[str],
    exp_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    pkl_path = root / f"{model}-scene-graph.pkl"
    by_image: Dict[str, Dict[str, Any]] = {}
    relation_forcing_available = pkl_path.exists()
    if relation_forcing_available:
        pkl_entries = pickle.load(pkl_path.open("rb"))
        by_image = {str(entry["img_id"]): entry for entry in pkl_entries}

    pred_sg_dir = root / "coco_results" / model / "injected_pred"

    attr_qids = {int(r["question_id"]) for r in sampled_attr}
    rel_qids = {int(r["question_id"]) for r in sampled_rel}

    rel_audit: List[Dict[str, Any]] = []
    seg_map_cache: Dict[str, Dict[str, int]] = {}

    out_rows: List[Dict[str, Any]] = []
    stats = defaultdict(int)

    for row in base_rows:
        qid = int(row["question_id"])
        row_out = dict(row)

        if qid in attr_qids:
            stats["attr_sampled"] += 1
            gen = generated_attr_answers.get(qid)
            if gen:
                row_out["model_answer"] = gen
                stats["attr_replaced"] += 1
            else:
                stats["attr_generation_missing"] += 1

        if qid in rel_qids:
            stats["rel_sampled"] += 1
            image_id = str(row.get("image_id"))
            gt_answer = str(row.get("gt_answer", "")).strip()
            debug = row.get("debug") or {}
            pred_subj = debug.get("pred_subject_seg_id")
            pred_obj = debug.get("pred_object_seg_id")

            audit = {
                "question_id": qid,
                "image_id": image_id,
                "gt_answer": gt_answer,
                "pred_subject_seg_id": pred_subj,
                "pred_object_seg_id": pred_obj,
                "inserted": False,
                "reason": None,
                "top3_predicates": [],
            }

            if not relation_forcing_available:
                audit["reason"] = "scene_graph_pkl_missing"
                rel_audit.append(audit)
                out_rows.append(row_out)
                continue

            if pred_subj is None or pred_obj is None:
                audit["reason"] = "subject_or_object_not_matched"
                rel_audit.append(audit)
                out_rows.append(row_out)
                continue

            if image_id not in by_image:
                audit["reason"] = "image_missing_in_pkl"
                rel_audit.append(audit)
                out_rows.append(row_out)
                continue

            if image_id not in seg_map_cache:
                pred_json = pred_sg_dir / f"scene-graph_{image_id}.json"
                if not pred_json.exists():
                    seg_map_cache[image_id] = {}
                else:
                    seg_map_cache[image_id] = build_seg_to_box_index(pred_json)

            seg_to_idx = seg_map_cache[image_id]
            s_idx = seg_to_idx.get(str(pred_subj))
            o_idx = seg_to_idx.get(str(pred_obj))
            if s_idx is None or o_idx is None:
                audit["reason"] = "matched_seg_to_box_index_failed"
                rel_audit.append(audit)
                out_rows.append(row_out)
                continue

            top3 = top3_predicates_for_pair(by_image[image_id], s_idx, o_idx, predicate_classes)
            audit["top3_predicates"] = top3
            if not top3:
                audit["reason"] = "pair_missing_in_pkl_pairs"
                rel_audit.append(audit)
                out_rows.append(row_out)
                continue

            gt_candidates = split_gt_relation_candidates(gt_answer)
            top3_norm = {normalize_text(p) for p in top3}
            matched = any(normalize_text(c) in top3_norm for c in gt_candidates)

            if matched:
                row_out["model_answer"] = gt_answer
                audit["inserted"] = True
                audit["reason"] = "gt_in_top3"
                stats["rel_replaced"] += 1
            else:
                audit["reason"] = "gt_not_in_top3"

            rel_audit.append(audit)

        out_rows.append(row_out)

    write_json(exp_dir / model / "relation_insertion_audit.json", rel_audit)
    return out_rows, rel_audit, dict(stats)


def filter_target_non_none(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("question_type") not in TARGET_TYPES:
            continue
        if is_none_answer(r.get("model_answer")):
            continue
        out.append(r)
    return out


def run_subprocess(cmd: List[str], cwd: Path, log_path: Optional[Path] = None) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write("=== STDOUT ===\n")
            fh.write(result.stdout or "")
            fh.write("\n=== STDERR ===\n")
            fh.write(result.stderr or "")
    if result.returncode != 0:
        print(
            f"[ERROR] Subprocess failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}",
            flush=True,
        )
        # Print last 10 000 chars of stderr so it appears in the Slurm log.
        stderr_tail = (result.stderr or "")[-10_000:]
        stdout_tail = (result.stdout or "")[-2_000:]
        if stderr_tail:
            print(f"[STDERR]\n{stderr_tail}", flush=True)
        if stdout_tail:
            print(f"[STDOUT]\n{stdout_tail}", flush=True)
        if log_path is not None:
            print(f"[INFO] Full subprocess output written to {log_path}", flush=True)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    # Echo child stdout for visibility.
    if result.stdout:
        print(result.stdout, flush=True)


def load_scored_map(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in rows:
        qid = r.get("question_id")
        if qid is None:
            qid = r.get("qid")
        if qid is None:
            continue
        out[int(qid)] = r
    return out


def filter_scored_by_questions(
    scored_rows: List[Dict[str, Any]],
    valid_qids: set[int],
) -> List[Dict[str, Any]]:
    out = []
    for r in scored_rows:
        qid = r.get("question_id")
        if qid is None:
            qid = r.get("qid")
        if qid is None:
            continue
        if int(qid) in valid_qids:
            out.append(r)
    return out


def extract_metrics(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {
        "ccta": data.get("ccta", 0.0),
        "aw_ccta": data.get("aw_ccta", 0.0),
        "cir": data.get("consistency_integrity_ratio", 0.0),
        "num_facts": data.get("num_facts", 0),
        "by_question_type": data.get("by_question_type", {}),
    }
    return out


def sample_qids(candidates: List[int], k: int, rng: random.Random) -> List[int]:
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)
    return rng.sample(candidates, k)


def build_scored_rows_for_qids(
    scored_by_qid: Dict[int, Dict[str, Any]],
    qids: List[int],
) -> List[Dict[str, Any]]:
    out = []
    for qid in qids:
        row = scored_by_qid.get(int(qid))
        if row is not None:
            out.append(row)
    return out


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_name = args.run_name or f"forced_none_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = root / "experiment_outputs" / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    predicate_classes = load_predicate_classes(root)
    attribute_model = args.attribute_model or args.judge_model
    llm = None

    summary: Dict[str, Any] = {
        "run_name": run_name,
        "seed": args.seed,
        "samples_per_type": args.samples_per_type,
        "judge_model": args.judge_model,
        "num_gpus": args.num_gpus,
        "models": {},
    }

    for model in args.models:
        print(f"\n=== Processing {model} ===")
        model_dir = root / "coco_results" / model
        ige_answers_path = model_dir / f"ige_answers_{model}.jsonl"
        base_rows = load_jsonl(ige_answers_path)

        sampled_attr, attr_shortfall = sample_none_rows(base_rows, ATTR_TYPE, args.samples_per_type, rng)
        sampled_rel, rel_shortfall = sample_none_rows(base_rows, REL_TYPE, args.samples_per_type, rng)
        attr_qids = {int(r["question_id"]) for r in sampled_attr}
        rel_qids = {int(r["question_id"]) for r in sampled_rel}

        model_exp = exp_dir / model
        write_json(model_exp / "sample_manifest.json", {
            "model": model,
            "attr_sampled": len(sampled_attr),
            "attr_shortfall": attr_shortfall,
            "rel_sampled": len(sampled_rel),
            "rel_shortfall": rel_shortfall,
            "attr_question_ids": [int(r["question_id"]) for r in sampled_attr],
            "rel_question_ids": [int(r["question_id"]) for r in sampled_rel],
        })

        image_dir = root / "coco_images" / model
        if args.skip_attribute_generation:
            generated_attr_answers = {}
            attr_meta = [
                {
                    "question_id": int(r["question_id"]),
                    "image_id": str(r["image_id"]),
                    "status": "skipped_by_flag",
                }
                for r in sampled_attr
            ]
        else:
            if llm is None:
                print(f"Initializing Qwen VL model for attribute generation: {attribute_model}")
                llm = init_qwen_llm(attribute_model, args.num_gpus, args.gpu_mem_util, args.max_model_len)
            generated_attr_answers, attr_meta = generate_attribute_answers(
                llm=llm,
                model_name=model,
                attr_rows=sampled_attr,
                image_dir=image_dir,
                max_tokens=args.max_gen_tokens,
                temperature=args.temperature,
            )
        write_json(model_exp / "attribute_generation_audit.json", attr_meta)

        # Free GPU VRAM so the judge subprocess can load its own model copy.
        destroy_llm(llm)
        llm = None

        augmented_rows, rel_audit, replace_stats = build_augmented_rows(
            model=model,
            root=root,
            base_rows=base_rows,
            sampled_attr=sampled_attr,
            sampled_rel=sampled_rel,
            generated_attr_answers=generated_attr_answers,
            predicate_classes=predicate_classes,
            exp_dir=exp_dir,
        )

        aug_answers_path = model_exp / f"ige_answers_{model}_augmented.jsonl"
        write_jsonl(aug_answers_path, augmented_rows)

        # Filter baseline/augmented answer rows for judge target.
        baseline_filtered_answers = filter_target_non_none(base_rows)
        augmented_filtered_answers = filter_target_non_none(augmented_rows)

        baseline_filtered_answers_path = model_exp / f"ige_answers_{model}_baseline_filtered.jsonl"
        augmented_filtered_answers_path = model_exp / f"ige_answers_{model}_augmented_filtered.jsonl"
        write_jsonl(baseline_filtered_answers_path, baseline_filtered_answers)
        write_jsonl(augmented_filtered_answers_path, augmented_filtered_answers)

        # Judge augmented filtered set with existing llm_judge.
        aug_scored_path = model_exp / f"ige_scored_{model}_augmented_filtered.jsonl"
        aug_metrics_path = model_exp / f"ige_metrics_{model}_augmented_filtered.json"
        if not args.skip_judge:
            run_subprocess(
                [
                    "python",
                    "VQA/llm_judge.py",
                    "--pred_jsonl",
                    str(augmented_filtered_answers_path),
                    "--out_scored_jsonl",
                    str(aug_scored_path),
                    "--out_metrics",
                    str(aug_metrics_path),
                    "--model",
                    args.judge_model,
                    "--num_gpus",
                    str(args.num_gpus),
                    "--mode",
                    "local",
                ],
                cwd=root,
                log_path=model_exp / "subprocess_logs" / "llm_judge_augmented.log",
            )

        # Build baseline filtered scored sets from existing scored files.
        vqa_scored_default = load_jsonl(root / "coco_outputs_VQA" / model / f"scored_{model}.jsonl")
        ige_scored_default = load_jsonl(root / "coco_results" / model / f"ige_scored_{model}.jsonl")

        baseline_qids = {int(r["question_id"]) for r in baseline_filtered_answers}
        augmented_qids = {int(r["question_id"]) for r in augmented_filtered_answers}

        baseline_vqa_scored = filter_scored_by_questions(vqa_scored_default, baseline_qids)
        baseline_ige_scored = filter_scored_by_questions(ige_scored_default, baseline_qids)
        augmented_vqa_scored = filter_scored_by_questions(vqa_scored_default, augmented_qids)

        baseline_vqa_path = model_exp / f"vqa_scored_{model}_baseline_filtered.jsonl"
        baseline_ige_path = model_exp / f"ige_scored_{model}_baseline_filtered.jsonl"
        augmented_vqa_path = model_exp / f"vqa_scored_{model}_augmented_filtered.jsonl"

        write_jsonl(baseline_vqa_path, baseline_vqa_scored)
        write_jsonl(baseline_ige_path, baseline_ige_scored)
        write_jsonl(augmented_vqa_path, augmented_vqa_scored)

        # Consistency computations.
        baseline_consistency_path = model_exp / f"aw_ccta_{model}_baseline_filtered.json"
        augmented_consistency_path = model_exp / f"aw_ccta_{model}_augmented_filtered.json"

        run_subprocess(
            [
                "python",
                "pipeline/scripts/consistency_agreement.py",
                "--vqa-scored",
                str(baseline_vqa_path),
                "--ige-scored",
                str(baseline_ige_path),
                "--threshold",
                str(args.threshold),
                "--output",
                str(baseline_consistency_path),
            ],
            cwd=root,
            log_path=model_exp / "subprocess_logs" / "consistency_baseline.log",
        )

        if not args.skip_judge:
            run_subprocess(
                [
                    "python",
                    "pipeline/scripts/consistency_agreement.py",
                    "--vqa-scored",
                    str(augmented_vqa_path),
                    "--ige-scored",
                    str(aug_scored_path),
                    "--threshold",
                    str(args.threshold),
                    "--output",
                    str(augmented_consistency_path),
                ],
                cwd=root,
                log_path=model_exp / "subprocess_logs" / "consistency_augmented.log",
            )

        # Requested subset experiments:
        # 1) default_200: random from baseline non-None target set
        # 2) forced_200: random from sampled forced rows that now have augmented scores
        # 3) combined_400: union of default_200 and forced_200
        subset_metrics: Dict[str, Any] = {}
        default_pool = sorted(list(baseline_qids))
        forced_seed_pool = sorted(list((attr_qids | rel_qids) & augmented_qids))

        default_qids_sample = sample_qids(default_pool, args.default_set_samples, rng)
        forced_qids_sample = sample_qids(forced_seed_pool, args.forced_set_samples, rng)
        combined_qids = sorted(set(default_qids_sample) | set(forced_qids_sample))

        baseline_ige_by_qid = load_scored_map(baseline_ige_scored)
        baseline_vqa_by_qid = load_scored_map(baseline_vqa_scored)
        augmented_ige_rows = load_jsonl(aug_scored_path) if (not args.skip_judge and aug_scored_path.exists()) else []
        augmented_ige_by_qid = load_scored_map(augmented_ige_rows)
        augmented_vqa_by_qid = load_scored_map(augmented_vqa_scored)

        # default_200: baseline vs baseline
        default200_vqa_rows = build_scored_rows_for_qids(baseline_vqa_by_qid, default_qids_sample)
        default200_ige_rows = build_scored_rows_for_qids(baseline_ige_by_qid, default_qids_sample)
        default200_vqa_path = model_exp / f"vqa_scored_{model}_default200.jsonl"
        default200_ige_path = model_exp / f"ige_scored_{model}_default200.jsonl"
        default200_metrics_path = model_exp / f"aw_ccta_{model}_default200.json"
        write_jsonl(default200_vqa_path, default200_vqa_rows)
        write_jsonl(default200_ige_path, default200_ige_rows)
        run_subprocess(
            [
                "python",
                "pipeline/scripts/consistency_agreement.py",
                "--vqa-scored",
                str(default200_vqa_path),
                "--ige-scored",
                str(default200_ige_path),
                "--threshold",
                str(args.threshold),
                "--output",
                str(default200_metrics_path),
            ],
            cwd=root,
            log_path=model_exp / "subprocess_logs" / "consistency_default200.log",
        )
        subset_metrics["default_200"] = {
            "requested": args.default_set_samples,
            "actual": len(default_qids_sample),
            "metrics": extract_metrics(default200_metrics_path),
            "paths": {
                "vqa_scored": str(default200_vqa_path),
                "ige_scored": str(default200_ige_path),
                "consistency": str(default200_metrics_path),
            },
        }

        if not args.skip_judge:
            # forced_200: baseline VQA vs augmented IGE for forced qids
            forced200_vqa_rows = build_scored_rows_for_qids(augmented_vqa_by_qid, forced_qids_sample)
            forced200_ige_rows = build_scored_rows_for_qids(augmented_ige_by_qid, forced_qids_sample)
            forced200_vqa_path = model_exp / f"vqa_scored_{model}_forced200.jsonl"
            forced200_ige_path = model_exp / f"ige_scored_{model}_forced200.jsonl"
            forced200_metrics_path = model_exp / f"aw_ccta_{model}_forced200.json"
            write_jsonl(forced200_vqa_path, forced200_vqa_rows)
            write_jsonl(forced200_ige_path, forced200_ige_rows)
            run_subprocess(
                [
                    "python",
                    "pipeline/scripts/consistency_agreement.py",
                    "--vqa-scored",
                    str(forced200_vqa_path),
                    "--ige-scored",
                    str(forced200_ige_path),
                    "--threshold",
                    str(args.threshold),
                    "--output",
                    str(forced200_metrics_path),
                ],
                cwd=root,
                log_path=model_exp / "subprocess_logs" / "consistency_forced200.log",
            )
            subset_metrics["forced_200"] = {
                "requested": args.forced_set_samples,
                "actual": len(forced_qids_sample),
                "metrics": extract_metrics(forced200_metrics_path),
                "paths": {
                    "vqa_scored": str(forced200_vqa_path),
                    "ige_scored": str(forced200_ige_path),
                    "consistency": str(forced200_metrics_path),
                },
            }

            # combined_400: default_200 (baseline IGE) + forced_200 (augmented IGE)
            combined_vqa_rows = build_scored_rows_for_qids(augmented_vqa_by_qid, combined_qids)
            combined_ige_rows: List[Dict[str, Any]] = []
            forced_qid_set = set(forced_qids_sample)
            for qid in combined_qids:
                source_map = augmented_ige_by_qid if qid in forced_qid_set else baseline_ige_by_qid
                row = source_map.get(int(qid))
                if row is not None:
                    combined_ige_rows.append(row)
            combined_vqa_path = model_exp / f"vqa_scored_{model}_combined400.jsonl"
            combined_ige_path = model_exp / f"ige_scored_{model}_combined400.jsonl"
            combined_metrics_path = model_exp / f"aw_ccta_{model}_combined400.json"
            write_jsonl(combined_vqa_path, combined_vqa_rows)
            write_jsonl(combined_ige_path, combined_ige_rows)
            run_subprocess(
                [
                    "python",
                    "pipeline/scripts/consistency_agreement.py",
                    "--vqa-scored",
                    str(combined_vqa_path),
                    "--ige-scored",
                    str(combined_ige_path),
                    "--threshold",
                    str(args.threshold),
                    "--output",
                    str(combined_metrics_path),
                ],
                cwd=root,
                log_path=model_exp / "subprocess_logs" / "consistency_combined400.log",
            )
            subset_metrics["combined_400"] = {
                "requested": args.default_set_samples + args.forced_set_samples,
                "actual": len(combined_qids),
                "metrics": extract_metrics(combined_metrics_path),
                "paths": {
                    "vqa_scored": str(combined_vqa_path),
                    "ige_scored": str(combined_ige_path),
                    "consistency": str(combined_metrics_path),
                },
            }

        base_metrics = extract_metrics(baseline_consistency_path)
        if not args.skip_judge:
            aug_metrics = extract_metrics(augmented_consistency_path)
            delta = {
                "ccta": round(aug_metrics["ccta"] - base_metrics["ccta"], 6),
                "aw_ccta": round(aug_metrics["aw_ccta"] - base_metrics["aw_ccta"], 6),
                "cir": round(aug_metrics["cir"] - base_metrics["cir"], 6),
                "num_facts": int(aug_metrics["num_facts"] - base_metrics["num_facts"]),
            }
        else:
            aug_metrics = {}
            delta = {}

        summary["models"][model] = {
            "sampled": {
                "attr": len(sampled_attr),
                "rel": len(sampled_rel),
                "attr_shortfall": attr_shortfall,
                "rel_shortfall": rel_shortfall,
            },
            "attribute_generation": {
                "generated_count": len(generated_attr_answers),
            },
            "relation_insertion": {
                "inserted_count": sum(1 for r in rel_audit if r.get("inserted")),
                "audited_count": len(rel_audit),
            },
            "replacement_stats": replace_stats,
            "baseline": base_metrics,
            "augmented": aug_metrics,
            "delta": delta,
            "paths": {
                "augmented_answers": str(aug_answers_path),
                "augmented_scored": str(aug_scored_path),
                "baseline_consistency": str(baseline_consistency_path),
                "augmented_consistency": str(augmented_consistency_path),
            },
            "subset_experiments": subset_metrics,
        }

    write_json(exp_dir / "summary.json", summary)
    print("\nExperiment complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL] Unhandled exception in main():", flush=True)
        traceback.print_exc()
        raise
