#!/usr/bin/env python3
"""
CLI caller script for multi-model image generation.

Reads prompts produced by full_workflow (e.g. 8_prompts_gt/prompts.json) and
generates images for a fixed set of UniVLM-supported models.

Usage:
    python call_generate_images.py \
        --prompts-json /path/to/8_prompts_gt/prompts.json \
        --output-dir /path/to/gt_run/9_generated_images
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from generate_images_all_models import generate_images_for_all_models


FIXED_MODELS: List[str] = ["mmada", "blip3o", "showo2", "showo", "januspro", "omnigen2", "unitok", "tar", "bagel"]


def _load_prompts_from_full_workflow(prompts_json_path: Path) -> List[Tuple[str, str]]:
    """
    full_workflow Step A writes prompts.json as a list of objects:
    [{"image_id": ..., "prompt": "..."}, ...]
    We convert it into [(image_id, caption), ...] for the UniVLM generator.
    """
    with prompts_json_path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, dict):
        # Convert dict {"id": {...}, ...} to list [{...}, ...]
        data = list(data.values())

    if not isinstance(data, list):
        raise ValueError(
            f"Expected full_workflow prompts.json to be a JSON list or dict, got {type(data)}"
        )

    prompts: List[Tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id", ""))
        caption = item.get("prompt") or item.get("caption") or item.get("text") or ""
        caption = str(caption).strip()
        if caption and image_id:
            prompts.append((image_id, caption))

    # Sort by numeric image_id when possible (helps reproducibility)
    def _sort_key(t: Tuple[str, str]):
        try:
            return (0, int(t[0]))
        except Exception:
            return (1, t[0])

    prompts.sort(key=_sort_key)
    return prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images from full_workflow prompts.json (multi-model).")
    parser.add_argument(
        "--prompts-json",
        required=True,
        help="Path to full_workflow prompts.json (e.g. <gt_run_dir>/8_prompts_gt/prompts.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated images (will create subdirs per model)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (default: 0)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (kept for API compatibility)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (default: 42)")
    parser.add_argument("--models", nargs="+", default=FIXED_MODELS, help="Specific models to generate images for (default: all 6 models)")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging")
    
    # Splitting arguments
    parser.add_argument("--split-index", type=int, default=0, help="Index of the current split (0-based)")
    parser.add_argument("--num-splits", type=int, default=1, help="Total number of splits")

    args = parser.parse_args()

    prompts_path = Path(args.prompts_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not prompts_path.exists():
        raise SystemExit(f"ERROR: prompts.json not found: {prompts_path}")

    prompts = _load_prompts_from_full_workflow(prompts_path)
    if not prompts:
        raise SystemExit(f"ERROR: No usable prompts found in: {prompts_path}")

    # Handle Splitting
    if args.num_splits > 1:
        import math
        total_prompts = len(prompts)
        chunk_size = math.ceil(total_prompts / args.num_splits)
        start_idx = args.split_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_prompts)
        
        prompts = prompts[start_idx:end_idx]
        print(f"Processing Split {args.split_index + 1}/{args.num_splits}: Prompts {start_idx} to {end_idx} (Count: {len(prompts)})")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    results = generate_images_for_all_models(
        prompts=prompts,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        models=args.models,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print(f"\nGeneration completed!")
    print(f"Prompts: {results.get('num_prompts')}")
    print(f"Output dir: {results.get('output_dir')}")
    print(f"Successful models: {results.get('successful_models')}/{results.get('num_models')}")

    for result in results.get("results", []):
        if result.get("success"):
            print(f"✓ {result.get('model_name')}: {result.get('num_generated')} images")
        else:
            print(f"✗ {result.get('model_name')}: {result.get('error')}")
