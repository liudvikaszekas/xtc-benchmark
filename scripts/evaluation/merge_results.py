import json
import argparse
import glob
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_index", required=True, nargs='+', help="gt_index.jsonl file(s)")
    parser.add_argument("--pred_files", required=True, nargs='+', help="model prediction jsonl files")
    parser.add_argument("--output", required=True, help="merged output jsonl")
    args = parser.parse_args()

    # Load GT
    gt_map = {}
    gt_files = []
    for g in args.gt_index:
        gt_files.extend(glob.glob(g))

    for gt_file in gt_files:
        chunk_dir = os.path.basename(os.path.dirname(gt_file))
        with open(gt_file, "r") as f:
            for line in f:
                item = json.loads(line)
                gt_map[(chunk_dir, item["qid"])] = item

    # Load Preds
    merged_map = {}
    
    # Expand globs if needed (shells usually do this, but just in case)
    files = []
    for p in args.pred_files:
        files.extend(glob.glob(p))
    
    for p_file in files:
        chunk_dir = os.path.basename(os.path.dirname(p_file))
        with open(p_file, "r") as f:
            for line in f:
                pred = json.loads(line)
                qid = pred.get("question_id")
                if qid is None:
                    qid = pred.get("qid")
                
                if qid is not None and (chunk_dir, qid) in gt_map:
                    entry = gt_map[(chunk_dir, qid)].copy()
                    
                    # Store temporary keys for sorting
                    entry["_chunk_dir"] = chunk_dir
                    entry["_orig_qid"] = qid

                    if "model_answer" in pred:
                        entry["model_answer"] = pred["model_answer"]
                    elif "answer" in pred:
                        entry["model_answer"] = pred["answer"]
                    elif "error" in pred:
                        entry["model_answer"] = f"ERROR: {pred['error']}"
                    else:
                        entry["model_answer"] = "ERROR: No answer found"
                        
                    merged_map[(chunk_dir, qid)] = entry
                else:
                    print(f"Warning: qid {qid} in chunk {chunk_dir} not found in GT or valid")

    merged = list(merged_map.values())

    # Helper function to extract integer chunk index
    def get_chunk_idx(c_dir):
        try:
            return int(c_dir.split('_')[-1])
        except ValueError:
            return 999999

    # Sort merged list by chunk index, then by original qid
    merged.sort(key=lambda x: (get_chunk_idx(x["_chunk_dir"]), x.get("_orig_qid", 0)))

    # Reassign global qids starting from 0 (or 1), maintaining the original dataset order
    # run_vqa_benchmark.py enumerate(qs) started at 0, so we start at 0
    for idx, item in enumerate(merged):
        item["qid"] = idx
        item["question_id"] = idx
        
        # Remove temporary sorting keys
        item.pop("_chunk_dir", None)
        item.pop("_orig_qid", None)

    with open(args.output, "w") as f:
        for item in merged:
            f.write(json.dumps(item) + "\n")
    
    print(f"Merged {len(merged)} records to {args.output}")

if __name__ == "__main__":
    main()
