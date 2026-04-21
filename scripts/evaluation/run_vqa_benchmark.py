#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../vlm-benchmark
import sys
sys.path.insert(0, str(REPO_ROOT))

# Add univlm/evaluation to sys.path for roundtrip_factory import
def _find_univlm_eval():
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent.parent / "submodules" / "univlm" / "evaluation",               # benchmark/submodules/univlm/evaluation (preferred)
        script_dir.parent.parent.parent.parent.parent / "univlm" / "evaluation",  # projects/univlm/evaluation
        script_dir.parent.parent.parent.parent / "univlm" / "evaluation",          # vlm-benchmark/../univlm/evaluation
        script_dir.parent.parent.parent / "univlm" / "evaluation",                 # vlm-benchmark/univlm/evaluation
    ]
    for p in possible_paths:
        if (p / "roundtrip_factory.py").exists():
            sys.path.insert(0, str(p.resolve()))
            return
    # Fallback: try the most likely default
    fallback = script_dir.parent.parent / "submodules" / "univlm" / "evaluation"
    sys.path.insert(0, str(fallback.resolve()))

_find_univlm_eval()

from benchmark.scripts.evaluation.univlm_eval.answer_image_questions_all_models import answer_questions_for_all_models
from benchmark.scripts.evaluation.univlm_eval.answer_image_questions_all_models import answer_questions_for_model_multi_gpu
import torch



def load_generated_questions(path: str):
    with open(path, "r") as f:
        return json.load(f)["questions"]


def load_possible_relations(path: str) -> list:
    """Load and flatten relations from custom_psg.json or relations.json."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Check if it's custom_psg.json format (has predicate_classes)
        if 'predicate_classes' in data:
            return data['predicate_classes']
        
        # Otherwise assume it's the old relations.json format (categorized)
        rels = set()
        for cat, items in data.items():
            if isinstance(items, list):
                rels.update(items)
        return sorted(list(rels))
    except Exception as e:
        print(f"Warning: Error loading relations: {e}")
        return []


def normalize_img_name(q: dict) -> str:
    """
    Map question record -> actual filename under --image_dir.
    
    Tries multiple formats:
    1. Unpadded (natural): 1000.jpg, 88040.jpg (what VLM-generated images use)
    2. Zero-padded COCO: 000000001000.jpg (12 digits)
    
    Returns the unpadded version first - that's what generated images use.
    """
    if q.get("image_id") is not None:
        img_id = int(q['image_id'])
        # Return unpadded first (natural) - generated images use this format
        return f"{img_id}.jpg"
    # fallback to whatever is stored (basename only)
    return os.path.basename(q["image_file"].replace("\\", "/"))


def candidate_image_names(q: dict) -> list:
    names = []

    if q.get("image_id") is not None:
        try:
            img_id = str(int(q["image_id"]))
        except Exception:
            img_id = str(q["image_id"])
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            names.append(f"{img_id}{ext}")

    if q.get("image_file"):
        base = os.path.basename(str(q["image_file"]).replace("\\", "/"))
        stem, ext = os.path.splitext(base)
        if ext:
            names.append(base)
        if stem:
            for e in (".jpg", ".jpeg", ".png", ".webp"):
                names.append(f"{stem}{e}")

    # Deduplicate while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_questions", required=True, help="output/generated_questions.json")
    ap.add_argument("--image_dir", required=True, help="folder containing images referenced by questions")
    ap.add_argument("--output_dir", required=True, help="where UniVLM jsonl outputs go")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, help="Specific GPU IDs to use (e.g. 0 1)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--chunk_index", type=int, default=0, help="Index of the dataset chunk to process (0-indexed)")
    ap.add_argument("--num_chunks", type=int, default=1, help="Total number of chunks to split the dataset into")
    ap.add_argument("--shared_model_config", type=str, help="JSON string of shared model config")
    
    # Default relations file path - use custom_psg.json from output_kmax
    default_relations = str(REPO_ROOT / "benchmark" / "configs" / "custom_psg.json")
    ap.add_argument("--relations_file", default=default_relations, help="Path to custom_psg.json with predicate_classes")
    ap.add_argument("--skip_relations_context", action="store_true", help="Skip adding the list of possible relations to the prompt")
    
    args = ap.parse_args()

    qs = load_generated_questions(args.generated_questions)

    # Handle dataset chunking
    if args.num_chunks > 1:
        if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
            raise ValueError(f"--chunk_index {args.chunk_index} must be between 0 and {args.num_chunks - 1}")
        chunk_size = len(qs) // args.num_chunks
        remainder = len(qs) % args.num_chunks
        
        start_idx = args.chunk_index * chunk_size + min(args.chunk_index, remainder)
        end_idx = start_idx + chunk_size + (1 if args.chunk_index < remainder else 0)
        qs = qs[start_idx:end_idx]
        print(f"Processing chunk {args.chunk_index + 1}/{args.num_chunks} with {len(qs)} questions (from {start_idx} to {end_idx - 1}).")

    if args.limit and args.limit > 0:
        qs = qs[:args.limit]
    
    # Load possible relations for context
    possible_relations = load_possible_relations(args.relations_file)
    if possible_relations:
        print(f"Loaded {len(possible_relations)} possible relations for VQA context.")
    else:
        print("Warning: No relations dictionary loaded.")

    # Build paired list: [(image_name, qid, prompt_text), ...]
    # prompt_text = original question + relations (for relationship questions) + instruction
    qa_pairs = []
    missing = []

    # Search both the provided image_dir and common nested model subdirs
    search_prefixes = [""]
    for m in args.models:
        model_prefix = m.strip()
        if model_prefix and os.path.isdir(os.path.join(args.image_dir, model_prefix)):
            search_prefixes.append(model_prefix)

    # Fallback: if there's exactly one image-containing subdir, include it
    if len(search_prefixes) == 1:
        try:
            subdirs = [d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d))]
            image_subdirs = []
            for d in subdirs:
                full = os.path.join(args.image_dir, d)
                if any(name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) for name in os.listdir(full)):
                    image_subdirs.append(d)
            if len(image_subdirs) == 1:
                search_prefixes.append(image_subdirs[0])
        except Exception:
            pass

    if len(search_prefixes) > 1:
        print(f"Image lookup prefixes: {search_prefixes}")

    for i, q in enumerate(qs):
        img_name = normalize_img_name(q)
        resolved_rel = None
        resolved_abs = None

        for cand in candidate_image_names(q):
            for prefix in search_prefixes:
                rel = os.path.join(prefix, cand) if prefix else cand
                abs_path = os.path.join(args.image_dir, rel)
                if os.path.exists(abs_path):
                    resolved_rel = rel
                    resolved_abs = abs_path
                    break
            if resolved_abs is not None:
                break

        if resolved_abs is None:
            missing.append(os.path.join(args.image_dir, img_name))
            continue  # Skip questions for missing images
        
        # Add instruction to prompt, but keep original question for logging
        prompt_text = q["question"]
        
        # For relationship questions, append the relations dictionary if not skipped
        # For relationship questions, append the relations dictionary if not skipped
        # REMOVED as per user request: "remove the option to include all relations in the prompt. This will never be used again."
        # if q.get("question_type") == "label_attributes_to_relationship" and possible_relations and not q.get("meta", {}).get("options") and not args.skip_relations_context:
        #     rels_str = ", ".join(possible_relations)
        #     prompt_text += f"\nChoose from the following relations: {rels_str}\nYou MUST choose exactly one of these options."
        
        # Determine instruction based on question type
        if q.get("question_type") == "label_attributes_to_relationship":
            prompt_text += " Answer with the matching relationship(s)."
        else:
            prompt_text += " Answer in no more than 3 words."
        
        qa_pairs.append((resolved_rel, i, prompt_text))

    if missing:
        print(f"Warning: {len(missing)} images not found. Skipping them.")
    else:
        print(f"All {len(qa_pairs)} questions have matching images.")

    # Write GT index for later evaluation/join
    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "gt_index.jsonl")
    with open(index_path, "w") as f:
        for i, q in enumerate(qs):
            rec = {
                "qid": i,
                "image_file": q["image_file"],
                "question": q["question"],
                "gt_answer": q["answer"],
                "template_index": q.get("template_index"),
                "question_type": q.get("question_type"),
                "attribute": q.get("meta", {}).get("attribute"),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote GT index: {index_path} ({len(qs)} rows)")

    # Load shared model config if provided
    shared_cfg = json.loads(args.shared_model_config) if args.shared_model_config else {}
    model_paths = shared_cfg.get("model_paths")
    config_paths = shared_cfg.get("config_paths")

    # Run UniVLM (paired, NOT cartesian product)
    if args.gpus:
        gpus = args.gpus
    else:
        gpus = list(range(torch.cuda.device_count()))
    
    # Or manually:
    # gpus = [0, 1]

    results_all = []
    for model_name in args.models:
        print(f"=== Running VQA for model: {model_name} ===")
        try:
            results = answer_questions_for_model_multi_gpu(
                image_dir=args.image_dir,
                qa_pairs=qa_pairs,
                model_type=model_name,     
                output_dir=args.output_dir,
                gpus=gpus,
                seed=args.seed,
                batch_size=args.batch_size,
                verbose=args.verbose,
                model_paths=model_paths,
                config_paths=config_paths,
            )
            results_all.append({model_name: results})
        except Exception as e:
            print(f"Error running model {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")
    print(json.dumps(results_all, indent=2))


if __name__ == "__main__":
    main()
