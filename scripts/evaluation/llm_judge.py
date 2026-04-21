#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess
import time
import pickle
from pathlib import Path
from typing import List, Dict

try:
    import torch
except ImportError:
    pass

try:
    from pydantic import BaseModel, Field
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_jsonl", required=True, nargs='+', help="Path to aggregated predictions file(s)")
    parser.add_argument("--out_metrics", required=False, help="Path to output metrics json (ignored in batch mode)")
    parser.add_argument("--out_scored_jsonl", required=False, help="Path to output scored jsonl (ignored in batch mode)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct", help="LLM Model to use as judge")
    
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for local inference")
    
    # Execution mode
    parser.add_argument("--mode", choices=["local", "slurm", "worker"], default="local", 
                        help="'local': run on current node. 'slurm': submit job. 'worker': internal use.")
    
    # Slurm configuration (optional if mode=slurm)
    parser.add_argument("--slurm_account", default="sci-zacharatou")
    parser.add_argument("--slurm_partition", default="gpu-batch")
    parser.add_argument("--slurm_gpus", type=int, default=1)
    parser.add_argument("--slurm_mem", default="64G")
    parser.add_argument("--slurm_time", default="01:00:00")
    
    # Worker args
    parser.add_argument("--input_pickle", help="For worker mode")
    parser.add_argument("--output_pickle", help="For worker mode")
    
    return parser.parse_args()

class JudgeScore(BaseModel):
    reasoning: str = Field(..., description="Brief analysis comparing the predicted answer to the ground truth.")
    score: int = Field(..., description="The integer score between 0 (completely wrong) and 5 (completely right).")

def construct_prompt(record: Dict) -> str:
    q = record.get("question", "")
    gt = str(record.get("gt_answer", ""))
    pred = str(record.get("model_answer", ""))
    
    # Using JSON structure prompt
    return (
        f"<|im_start|>system\nYou are a precise scoring assistant. Analyze the predicted answer against the correct answer and assign a score.\n"
        "Return your response in JSON format with two fields: 'reasoning' and 'score'.\n"
        "If the ground truth answer includes multiple options (separated by ',', 'and', or 'or'), score the predicted answer based on the best matching option from the ground truth.\n"
        "Scoring Scale:\n"
        "0 = completely wrong\n"
        "1 = almost completely wrong\n"
        "2 = mostly wrong\n"
        "3 = half right\n"
        "4 = mostly right\n"
        "5 = completely right\n<|im_end|>\n"
        f"<|im_start|>user\n"
        "Question: What color is the sky?\n"
        "Correct: blue\n"
        "Predicted: Blue.\n"
        "Response: {{\"reasoning\": \"Exact match.\", \"score\": 5}}\n\n"
        "Question: How many apples?\n"
        "Correct: 3\n"
        "Predicted: 2\n"
        "Response: {{\"reasoning\": \"Incorrect number, but close.\", \"score\": 0}}\n\n"
        "What is the spatial relationship between the sky and the car?"
        "Correct: above"
        "Predicted: above car"
        "Response: {{\"reasoning\": \"The sky is above the car.\", \"score\": 5}}\n\n"
        "How is the truck moving through space relative to the pavement?"
        "Correct: driving on and parked on"
        "Predicted: parked"
        "Response: {{\"reasoning\": \"Parked matches one of the correct answers.\", \"score\": 5}}\n\n"
        f"Question: {q}\n"
        f"Correct: {gt}\n"
        f"Predicted: {pred}\n"
        "Response:\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def load_vllm_model(model_name: str, num_gpus: int = 1):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vllm not installed.")
        sys.exit(1)

    print(f"Loading {model_name} on {num_gpus} GPUs...")
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=num_gpus, 
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        enforce_eager=True
    )
    return llm

def generate_scores_with_llm(llm, records: List[Dict]) -> List[int]:
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        GuidedDecodingParams = None
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        StructuredOutputsParams = None

    # Use schema-constrained decoding to enforce JudgeScore output format.
    if GuidedDecodingParams:
        guided_params = GuidedDecodingParams(json=JudgeScore.model_json_schema())
        sampling_params = SamplingParams(
            max_tokens=256, # Increased for reasoning
            temperature=0.0,
            guided_decoding=guided_params
        )
    elif StructuredOutputsParams:
        sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(
                json=JudgeScore.model_json_schema(),
                disable_additional_properties=True,
            )
        )
    else:
        print("WARNING: No schema decoding API available (GuidedDecodingParams/StructuredOutputsParams). Using unconstrained sampling.")
        sampling_params = SamplingParams(max_tokens=256, temperature=0.0)
    
    prompts = [construct_prompt(r) for r in records]
    print(f"Generating scores for {len(prompts)} records with structured JSON output...")
    
    outputs = llm.generate(prompts, sampling_params)
    
    scores = []
    failed_count = 0
    
    # Debug: print first 5 outputs
    for i in range(min(5, len(outputs))):
        text = outputs[i].outputs[0].text.strip()
        gt = records[i].get('gt_answer', '')
        pred = records[i].get('model_answer', '')
        print(f"Record {i}: GT='{gt}' | Pred='{pred}' | Output='{text}'")
        
    for o in outputs:
        text = o.outputs[0].text.strip()
        try:
            # Parse the enforced JSON
            data = json.loads(text)
            score = data.get("score")
            
            # Additional safety check
            if isinstance(score, int) and 0 <= score <= 5:
                scores.append(score)
            else:
                print(f"ERROR: Score {score} out of range!")
                scores.append(None)
                failed_count += 1
        except json.JSONDecodeError:
            print(f"ERROR: Could not parse JSON: '{text}'")
            scores.append(None)
            failed_count += 1
        except Exception as e:
            print(f"ERROR: Unexpected error parsing: {e}")
            scores.append(None)
            failed_count += 1
    
    print(f"Score generation complete. Failed: {failed_count}/{len(records)}")
    return scores

def process_single_file(input_file, llm, args, scores=None):
    print(f"Reading predictions from {input_file}...")
    records = []
    with open(input_file, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass
    if not records:
        print(f"Warning: No valid records found in {input_file}. Skipping.")
        return

    if args.mode == "local":
        if llm is None:
             raise ValueError("LLM not initialized")
        scores = generate_scores_with_llm(llm, records)
    elif args.mode == "slurm":
        scores = run_slurm_mode(args, records)
        
    # Merge scores back to records and compute metrics
    scored_records = []
    total_valid = 0
    total_score = 0
    correct_count = 0 
    failed_parse_count = 0
    
    from collections import defaultdict
    by_template = defaultdict(lambda: {"total": 0, "sum_score": 0, "failed": 0})
    
    for r, score in zip(records, scores):
        r["llm_judge_score"] = score
        
        if score is not None:
             r["score"] = score / 5.0 
             r["is_correct"] = (score >= 4)
             
             total_valid += 1
             total_score += score
             if score >= 4:
                correct_count += 1
        else:
             r["score"] = None
             r["is_correct"] = None
             failed_parse_count += 1
        
        ordered_r = {}
        priority_keys = ["question_id", "question_type", "question", "gt_answer", "model_answer", "llm_judge_score", "score", "is_correct"]
        for k in priority_keys:
            if k in r:
                ordered_r[k] = r[k]
        for k, v in r.items():
            if k not in ordered_r:
                ordered_r[k] = v
                
        scored_records.append(ordered_r)
        
        tid = str(r.get("template_index", "NA"))
        if score is not None:
            by_template[tid]["total"] += 1
            by_template[tid]["sum_score"] += score
        else:
            by_template[tid]["failed"] += 1

    # Metrics
    metrics = {
        "total_attempted": len(records),
        "total_scored": total_valid,
        "failed_parse_count": failed_parse_count,
        "average_score": total_score / total_valid if total_valid else 0,
        "accuracy_threshold_4": correct_count / total_valid if total_valid else 0,
        "judge_model": args.model,
        "by_template": {
            k: {
                "count_valid": v["total"],
                "count_failed": v["failed"],
                "average_score": v["sum_score"] / v["total"] if v["total"] else 0
            } for k, v in by_template.items()
        }
    }
    
    # Determine output paths
    if isinstance(args.pred_jsonl, list) and len(args.pred_jsonl) > 1:
        p = Path(input_file)
        stem = p.stem
        # Clean up stem if it ends with _merged
        if stem.endswith("_merged"):
            stem = stem[:-7]
            
        out_metrics = p.parent / f"metrics_{stem}.json"
        out_scored = p.parent / f"scored_{stem}.jsonl"
        print(f"Batch mode: Writing to {out_metrics} and {out_scored}")
    else:
        out_metrics = args.out_metrics
        out_scored = args.out_scored_jsonl

    print(f"Writing metrics to {out_metrics}")
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Writing scored lines to {out_scored}")
    with open(out_scored, "w") as f:
        for r in scored_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Done processing {input_file}.")
    print(f"Summary: {total_valid}/{len(records)} scored successfully. Failed: {failed_parse_count}")

def main():
    args = parse_args()
    
    # Handle worker mode first (simple case)
    if args.mode == "worker":
        # Worker mode
        print(f"Worker started. Loading inputs from {args.input_pickle}")
        with open(args.input_pickle, "rb") as f:
            data = pickle.load(f)
        records = data["records"]
        model = data.get("model", args.model)
        
        num_gpus = torch.cuda.device_count()
        llm = load_vllm_model(model, num_gpus)
        scores = generate_scores_with_llm(llm, records)
        
        print(f"Saving outputs to {args.output_pickle}")
        with open(args.output_pickle, "wb") as f:
            pickle.dump({"scores": scores}, f)
        return

    # Handle main mode (Local or Slurm orchestrator)
    input_files = args.pred_jsonl
    if isinstance(input_files, str):
        input_files = [input_files]
        
    llm = None
    if args.mode == "local":
        num_gpus = args.num_gpus
        if num_gpus <= 0:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1
        llm = load_vllm_model(args.model, num_gpus)
        
    for input_file in input_files:
        process_single_file(input_file, llm, args)

if __name__ == "__main__":
    main()
