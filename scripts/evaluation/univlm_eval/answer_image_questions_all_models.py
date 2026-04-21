#!/usr/bin/env python3
"""
Image Question Answering for All Models - Modular Version

This module answers questions about images using all supported models.
Can be imported and called from other Python scripts.

Usage as module:
    from answer_image_questions_all_models import answer_questions_for_all_models
    
    results = answer_questions_for_all_models(
        image_dir="/path/to/images",
        question="What is in this image?",
        output_dir="results",
        models=["blip3o", "mmada"],
        device=0
    )

Usage as script:
    python answer_image_questions_all_models.py --image_dir /path/to/images --question "What is this?"
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_base_storage() -> Path:
    base_storage = os.environ.get("UNIVLM_BASE_STORAGE") or os.environ.get("BASE_STORAGE")
    if base_storage:
        return Path(base_storage).expanduser().resolve()
    return _find_repo_root() / ".tmp" / "baselines"


def _get_models_dir() -> Path:
    models_dir = os.environ.get("MODELS_DIR")
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    return _get_base_storage() / "models"


def _resolve_showo2_config(univlm_path: Path) -> str:
    env_cfg = os.environ.get("UNIVLM_SHOWO2_CONFIG") or os.environ.get("SHOWO2_CONFIG_OUTPUT")
    if env_cfg:
        return str(Path(env_cfg).expanduser().resolve())

    generated_cfg = _get_base_storage() / "configs" / "showo2_config.yaml"
    if generated_cfg.exists():
        return str(generated_cfg)

    source_cfg = univlm_path / "configs" / "showo2_config.yaml"
    if not source_cfg.exists():
        return str(source_cfg)

    vae_candidates = [
        os.environ.get("UNIVLM_SHOWO2_VAE"),
        os.environ.get("SHOWO2_VAE_PATH"),
        str(_get_models_dir() / "Wan2.1_VAE.pth"),
        str(_get_base_storage() / "models" / "Wan2.1_VAE.pth"),
    ]
    valid_vae = next((Path(p).expanduser().resolve() for p in vae_candidates if p and Path(p).expanduser().exists()), None)
    if not valid_vae:
        return str(source_cfg)

    text = source_cfg.read_text(encoding="utf-8")
    current_match = re.search(r'^\s*pretrained_model_path:\s*"([^"]*Wan2\.1_VAE\.pth)"\s*$', text, flags=re.MULTILINE)
    if current_match and Path(current_match.group(1)).expanduser() == valid_vae:
        return str(source_cfg)

    patched = re.sub(
        r'^\s*pretrained_model_path:\s*"[^"]*Wan2\.1_VAE\.pth"\s*$',
        f'        pretrained_model_path: "{valid_vae}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    auto_cfg = _get_base_storage() / "configs" / "showo2_config.auto.yaml"
    auto_cfg.parent.mkdir(parents=True, exist_ok=True)
    auto_cfg.write_text(patched, encoding="utf-8")
    return str(auto_cfg)


def _ensure_blip_tokenizer_fallback() -> None:
    blip_path = UNIVLM_PATH / "BLIP3o"
    if str(blip_path) not in sys.path:
        sys.path.insert(0, str(blip_path))

    try:
        import blip3o.model.builder as blip_builder
    except Exception:
        return

    if getattr(blip_builder, "_benchmark_blip_tokenizer_patched", False):
        return

    orig_from_pretrained = blip_builder.AutoTokenizer.from_pretrained

    def _patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        try:
            return orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except KeyError as exc:
            if "blip3oQwenConfig" not in str(exc):
                raise
            return orig_from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", *args, **kwargs)

    blip_builder.AutoTokenizer.from_pretrained = _patched_from_pretrained
    blip_builder._benchmark_blip_tokenizer_patched = True

def _find_univlm():
    # Check common locations relative to this script
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent.parent.parent / "submodules" / "univlm",  # benchmark/submodules/univlm (preferred)
        script_dir.parent.parent / "univlm",         # home/univlm if script is moved under univlm_eval
        script_dir.parent.parent.parent / "univlm",  # home/univlm if script is moved under pipeline/scripts
        script_dir / "univlm",                       # local child
    ]
    for p in possible_paths:
        if (p / "evaluation" / "roundtrip_factory.py").exists():
            return p.resolve()
    # Fallback to the most likely default
    return (script_dir.parent.parent.parent / "submodules" / "univlm").resolve()

UNIVLM_PATH = _find_univlm()
MODELS_DIR = _get_models_dir()

sys.path.insert(0, str(UNIVLM_PATH / "evaluation"))

from roundtrip_factory import create_roundtrip_generator


# Model configurations with default paths
MODEL_CONFIGS = {
    "blip3o": {
        "name": "BLIP3o",
        "default_path": str(MODELS_DIR / "BLIP3o-Model-8B"),
        "requires_config": False,
    },
    "mmada": {
        "name": "MMaDA",
        "default_path": "Gen-Verse/MMaDA-8B-Base",
        "requires_config": False,
    },
    "emu3": {
        "name": "EMU3",
        "default_path": "BAAI/Emu3-Chat",
        "requires_config": False,
    },
    "omnigen2": {
        "name": "OmniGen2",
        "default_path": "OmniGen2/OmniGen2",
        "requires_config": False,
    },
    "januspro": {
        "name": "JanusPro",
        "default_path": "deepseek-ai/Janus-Pro-7B",
        "requires_config": False,
    },
    "showo2": {
        "name": "Show-o2",
        "default_path": "showlab/show-o2-7B",
        "requires_config": True,
        "default_config": _resolve_showo2_config(UNIVLM_PATH),
    },
    "showo": {
        "name": "Show-o",
        "default_path": "showlab/show-o",
        "requires_config": True,
        "default_config": str(UNIVLM_PATH / "configs" / "showo_config.yaml"),
    },
    "bagel": {
        "name": "Bagel",
        "default_path": "ByteDance-Seed/BAGEL-7B-MoT",
        "requires_config": False,
    },
    "tar": {
        "name": "Tar",
        "default_path": "csuhan/Tar-Lumina2",
        "requires_config": False,
    },
    "unitok": {
            "name": "UniTok",
            "default_path": "Alpha-VLLM/Lumina-mGPT-7B-512"
        },
}


def load_image_files(image_dir: str) -> List[Path]:
    """Load all image files from a directory."""
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(image_files)


def answer_questions_for_model(model_type: str, model_path: str, config_path: Optional[str],
                                image_files: List[Path], question: str,
                                output_dir: Path, device: int, seed: int, 
                                verbose: bool = True) -> Dict[str, Any]:
    """Answer questions about images for a single model."""
    result = {
        "model_type": model_type,
        "model_name": MODEL_CONFIGS[model_type]["name"],
        "success": False,
        "error": None,
        "answers": [],
        "num_answered": 0,
    }
    
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Answering with {MODEL_CONFIGS[model_type]['name']} ({model_type})")
            print(f"{'='*60}")
        
        # Create model-specific output directory
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize generator
        if verbose:
            print(f"Initializing {model_type} model...")
        generator = create_roundtrip_generator(
            model_type=model_type,
            model_path=model_path,
            device=device,
            seed=seed,
            config_path=config_path
        )
        if verbose:
            print(f"✓ Model initialized successfully")
        
        # Answer question for each image
        iterator = tqdm(image_files, desc=f"Answering with {model_type}") if verbose else image_files
        for idx, image_path in enumerate(iterator):
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Generate answer
                answer = generator.generate_caption_from_image(image, question)
                
                result["answers"].append({
                    "index": idx,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "answer": answer,
                })
                
            except Exception as e:
                if verbose:
                    print(f"  Error answering for image {idx}: {e}")
                result["answers"].append({
                    "index": idx,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "error": str(e),
                })
        
        # Save answers to text file
        answers_file = model_output_dir / "answers.txt"
        with open(answers_file, 'w', encoding='utf-8') as f:
            f.write(f"Question: {question}\n")
            f.write(f"{'='*60}\n\n")
            for answer_data in result["answers"]:
                f.write(f"Image: {answer_data['image_name']}\n")
                if "error" in answer_data:
                    f.write(f"Error: {answer_data['error']}\n")
                else:
                    f.write(f"Answer: {answer_data['answer']}\n")
                f.write(f"\n{'-'*60}\n\n")
        
        result["num_answered"] = len([a for a in result["answers"] if "error" not in a])
        result["success"] = result["num_answered"] > 0
        result["answers_file"] = str(answers_file)
        
        if verbose:
            print(f"✓ Answered {result['num_answered']}/{len(image_files)} questions successfully")
        
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"✗ Failed to answer questions with {model_type}: {e}")
    
    return result


from tqdm import tqdm

# ...
def _answer_for_single_model(
    model_type: str,
    model_path: str,
    config_path: Optional[str],
    image_dir: str,
    questions: List[tuple],   # [(image_name, question_id, question), ...]
    jsonl_path: Path,
    device: int,
    seed: int,
    batch_size: int,          # kept for API compatibility; not used in paired mode
    verbose: bool
) -> Dict[str, Any]:
    """Answer paired (image, question) items for a single model."""
    try:
        if verbose:
            print(f"\n[GPU {device}] {'='*60}")
            print(f"[GPU {device}] Processing {MODEL_CONFIGS[model_type]['name']} ({model_type})")
            print(f"[GPU {device}] {'='*60}")

        if model_type == "blip3o":
            _ensure_blip_tokenizer_fallback()

        generator = create_roundtrip_generator(
            model_type=model_type,
            model_path=model_path,
            device=device,
            seed=seed,
            config_path=config_path
        )

        num_answered = 0
        total = len(questions)

        pbar = tqdm(
            total=total,
            desc=f"[GPU {device}] {model_type}",
            unit="qa",
            disable=not verbose,
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        )

        image_dir_p = Path(image_dir)

        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for image_name, question_id, question in questions:
                img_path = image_dir_p / image_name
                try:
                    with Image.open(img_path) as im:
                        image = im.convert("RGB")

                    answer = generator.generate_caption_from_image(image, question)

                    jsonl_entry = {
                        "question_id": question_id,
                        "question": question,
                        "image_name": image_name,
                        "answer": answer,
                    }
                    num_answered += 1

                except Exception as e:
                    if verbose:
                        print(f"Error for {image_name}, Q{question_id}: {e}")
                    jsonl_entry = {
                        "question_id": question_id,
                        "question": question,
                        "image_name": image_name,
                        "error": str(e),
                    }

                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                pbar.update(1)

        pbar.close()

        return {
            "model_type": model_type,
            "model_name": MODEL_CONFIGS[model_type]["name"],
            "device": device,
            "success": True,
            "num_answered": num_answered,
            "jsonl_path": str(jsonl_path),
        }

    except Exception as e:
        return {
            "model_type": model_type,
            "model_name": MODEL_CONFIGS[model_type]["name"],
            "device": device,
            "success": False,
            "error": str(e),
        }


def answer_questions_for_all_models(
    image_dir: str,
    questions: List[tuple],  # [(image_name, question_id, question), ...]
    output_dir: str = "qa_results",
    models: Optional[List[str]] = None,
    model_paths: Optional[Dict[str, str]] = None,
    config_paths: Optional[Dict[str, str]] = None,
    device: int = 0,
    batch_size: int = 1,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Paired mode: answers exactly len(questions) items.
    (NO questions × all-images cartesian product.)
    """
    if models is None:
        models = list(MODEL_CONFIGS.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        print(f"Answering {len(questions)} (image, question) pairs")
        print(f"Using {len(models)} models: {', '.join(models)}")
        print(f"Device: GPU {device}")
        print(f"Output directory: {output_dir}")

    # Prepare model configs
    model_configs = []
    for model_type in models:
        model_config = MODEL_CONFIGS[model_type]

        model_path = model_paths[model_type] if (model_paths and model_type in model_paths) else model_config["default_path"]

        config_path = None
        if model_config.get("requires_config"):
            config_path = config_paths[model_type] if (config_paths and model_type in config_paths) else model_config.get("default_config")

        jsonl_path = output_dir / f"{model_type}_answers.jsonl"

        model_configs.append({
            "model_type": model_type,
            "model_path": model_path,
            "config_path": config_path,
            "jsonl_path": jsonl_path,
            "device": device,
        })

    all_results = []
    if verbose:
        print(f"\n⏩ Running {len(models)} models sequentially...")

    for cfg in model_configs:
        result = _answer_for_single_model(
            model_type=cfg["model_type"],
            model_path=cfg["model_path"],
            config_path=cfg["config_path"],
            image_dir=image_dir,
            questions=questions,
            jsonl_path=cfg["jsonl_path"],
            device=cfg["device"],
            seed=seed,
            batch_size=batch_size,
            verbose=verbose,
        )
        all_results.append(result)

    summary = {
        "image_dir": str(image_dir),
        "num_pairs": len(questions),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        "num_models": len(all_results),
        "successful_models": len([r for r in all_results if r["success"]]),
        "results": all_results,
    }

    metadata_path = output_dir / "qa_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {summary['successful_models']}/{summary['num_models']} models succeeded")
        print(f"Total paired items answered: {summary['num_pairs']}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")

    return summary


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Answer questions about images using all supported models with multi-GPU support"
    )
    
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--questions_json", type=str, required=True,
                       help="JSON file with list of [question_id, question] pairs")
    parser.add_argument("--output_dir", type=str, default="qa_results",
                       help="Output directory (default: qa_results)")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
                       help="Models to use (default: all)")
    parser.add_argument("--devices", type=int, nargs="+",
                       help="GPU device IDs to use (default: auto-detect all)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--no_multiprocessing", action="store_true",
                       help="Disable multiprocessing (run models sequentially)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output messages")
    
    args = parser.parse_args()
    
    # Load questions from JSON
    with open(args.questions_json, 'r') as f:
        questions = json.load(f)
    
    # Call the main function
    answer_questions_for_all_models(
        image_dir=args.image_dir,
        questions=questions,
        output_dir=args.output_dir,
        models=args.models,
        devices=args.devices,
        batch_size=args.batch_size,
        seed=args.seed,
        use_multiprocessing=not args.no_multiprocessing,
        verbose=not args.quiet,
    )

import math
import multiprocessing as mp
from pathlib import Path

def _chunk_list(xs, n_chunks):
    n = len(xs)
    if n == 0:
        return []
    if n_chunks <= 0:
        return [xs]
    chunk_size = math.ceil(n / n_chunks)
    return [xs[i:i+chunk_size] for i in range(0, n, chunk_size)]


def _worker_one_gpu(args):
    # Unpack worker args
    (model_type, model_path, config_path, image_dir, qa_pairs, jsonl_path, device, seed, batch_size, verbose) = args

    # IMPORTANT for CUDA + multiprocessing on many clusters:
    import torch
    torch.cuda.set_device(device)

    # Reuse your existing single-model paired worker
    return _answer_for_single_model(
        model_type=model_type,
        model_path=model_path,
        config_path=config_path,
        image_dir=image_dir,
        questions=qa_pairs,          # paired list shard
        jsonl_path=jsonl_path,
        device=device,
        seed=seed,
        batch_size=batch_size,
        verbose=verbose,
    )


def answer_questions_for_model_multi_gpu(
    image_dir: str,
    qa_pairs: list,                 # [(image_name, qid, question), ...]
    model_type: str,
    output_dir: str,
    gpus: list,                     # e.g. [0,1,2,3]
    seed: int = 42,
    batch_size: int = 1,
    verbose: bool = True,
    model_paths=None,
    config_paths=None,
):
    """
    Runs ONE model on multiple GPUs by sharding qa_pairs across GPUs.
    Writes one jsonl per GPU shard: <model>_answers.gpu<id>.jsonl
    Returns list of shard results.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_config = MODEL_CONFIGS[model_type]
    model_path = model_paths[model_type] if (model_paths and model_type in model_paths) else model_config["default_path"]
    config_path = None
    if model_config.get("requires_config"):
        config_path = config_paths[model_type] if (config_paths and model_type in config_paths) else model_config.get("default_config")

    if not qa_pairs:
        if verbose:
            print(f"No QA pairs provided for {model_type}; skipping multi-GPU run.")
        return []

    if not gpus:
        raise ValueError(f"No GPUs provided for multi-GPU run of model '{model_type}'.")

    shards = _chunk_list(qa_pairs, len(gpus))

    worker_args = []
    for gpu, shard in zip(gpus, shards):
        jsonl_path = outdir / f"{model_type}_answers.gpu{gpu}.jsonl"
        worker_args.append((
            model_type, model_path, config_path,
            image_dir, shard, jsonl_path,
            gpu, seed, batch_size, verbose
        ))

    if verbose:
        print(f"Running {model_type} on GPUs {gpus} with {len(qa_pairs)} pairs "
              f"({[len(s) for s in shards]} per GPU)")

    ctx = mp.get_context("spawn")  # safer with CUDA than fork
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_worker_one_gpu, worker_args)

    return results


if __name__ == "__main__":
    main()
