#!/usr/bin/env python3
"""
Image Generation for All Models - Modular Version

This module generates images from text prompts using all supported models.
Can be imported and called from other Python scripts.

Usage as module:
    from generate_images_all_models import generate_images_for_all_models
    
    results = generate_images_for_all_models(
        prompt="A beautiful sunset over mountains",
        output_dir="results",
        models=["blip3o", "mmada"],
        device=0
    )

Usage as script:
    python generate_images_all_models.py --prompt "A beautiful sunset" --output_dir results
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import torch

# Robust UNIVLM_PATH resolution
def _find_univlm():
    # Check common locations relative to this script
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent.parent / "submodules" / "univlm",   # benchmark/submodules/univlm (preferred)
        script_dir.parent.parent.parent.parent / "univlm",  # projects/univlm (sibling of benchmark repo)
        script_dir.parent.parent.parent / "univlm",  # home/univlm if script is moved under pipeline/scripts
        script_dir.parent.parent / "univlm",         # home/univlm if script is moved under univlm_eval
        script_dir / "univlm",                       # local child
    ]
    for p in possible_paths:
        if (p / "evaluation" / "roundtrip_factory.py").exists():
            return p.resolve()
    # Fallback to the most likely default
    return (script_dir.parent.parent / "submodules" / "univlm").resolve()

UNIVLM_PATH = _find_univlm()

sys.path.insert(0, str(UNIVLM_PATH / "evaluation"))

from roundtrip_factory import create_roundtrip_generator


# Model configurations with default paths
MODEL_CONFIGS = {
    "blip3o": {
        "name": "BLIP3o",
        "default_path": "~/models/BLIP3o-Model-8B",
        "requires_config": False,
    },
    "mmada": {
        "name": "MMaDA",
        "default_path": "Gen-Verse/MMaDA-8B-Base",
        "requires_config": False,
    },
    "emu3": {
        "name": "EMU3",
        "default_path": "BAAI/Emu3-Gen",
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
        "default_config": str(UNIVLM_PATH / "configs" / "showo2_config.yaml"),
    },
    "showo": {
        "name": "Show-o",
        "default_path": "showlab/show-o",
        "requires_config": True,
        "default_config": str(UNIVLM_PATH / "configs" / "showo_config.yaml"),
    },
    "tar": {
        "name": "TAR",
        "default_path": "csuhan/Tar-Lumina2",
        "requires_config": False,
    },
    "bagel": {
        "name": "Bagel",
        "default_path": "ByteDance-Seed/BAGEL-7B-MoT",
        "requires_config": False,
    },
}


def generate_images_for_model(model_type: str, model_path: str, config_path: Optional[str],
                               prompt: str, output_dir: Path, device: int, 
                               seed: int, verbose: bool = True) -> Dict[str, Any]:
    """Generate image for a single model from a single prompt."""
    result = {
        "model_type": model_type,
        "model_name": MODEL_CONFIGS[model_type]["name"],
        "success": False,
        "error": None,
        "image_path": None,
        "image_size": None,
    }
    
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating image with {MODEL_CONFIGS[model_type]['name']} ({model_type})")
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
        
        # Generate image
        if verbose:
            print(f"Generating image from prompt: {prompt[:100]}...")
        image = generator.generate_image_from_text(prompt, seed=seed)
        
        # Save image
        safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_')
        image_filename = f"generated_{safe_prompt}.jpg"
        image_path = model_output_dir / image_filename
        image.save(image_path)
        
        result["success"] = True
        result["image_path"] = str(image_path)
        result["image_size"] = image.size
        
        if verbose:
            print(f"✓ Generated image saved to: {image_path}")
        
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"✗ Failed to generate image with {model_type}: {e}")
    
    return result


def _generate_for_single_model(
    model_type: str,
    model_path: str,
    config_path: Optional[str],
    prompts: List[tuple],
    output_dir: Path,
    device: int,
    seed: int,
    batch_size: int,
    verbose: bool
) -> Dict[str, Any]:
    """Generate images for a single model."""
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {MODEL_CONFIGS[model_type]['name']} ({model_type})")
            print(f"{'='*60}")
        
        # Initialize generator
        generator = create_roundtrip_generator(
            model_type=model_type,
            model_path=model_path,
            device=device,
            seed=seed,
            config_path=config_path
        )
        
        # Process prompts
        model_results = []
        for image_id, caption in tqdm(prompts, desc=f"{MODEL_CONFIGS[model_type]['name']}", disable=not verbose):
            try:
                image_filename = f"{image_id}.jpg"
                image_path = output_dir / image_filename
                
                # Skip if image already exists
                if image_path.exists():
                    model_results.append({
                        "image_id": image_id,
                        "caption": caption,
                        "image_path": str(image_path),
                        "success": True,
                        "skipped": True
                    })
                    continue

                # OmniGen2 limit: 1024 tokens. Truncate/skip long prompts.
                if model_type == "omnigen2" and len(caption) > 2000:
                    error_msg = f"Skipping prompt (length {len(caption)} chars > 2000 for OmniGen2 limit)"
                    if verbose:
                        print(f"[{image_id}] {error_msg}")
                    model_results.append({
                        "image_id": image_id,
                        "caption": caption,
                        "error": error_msg,
                        "success": False
                    })
                    continue

                image = generator.generate_image_from_text(caption, seed=seed + int(image_id))
                
                # Save image
                image.save(image_path)
                
                model_results.append({
                    "image_id": image_id,
                    "caption": caption,
                    "image_path": str(image_path),
                    "success": True
                })
                
            except Exception as e:
                model_results.append({
                    "image_id": image_id,
                    "caption": caption,
                    "error": str(e),
                    "success": False
                })
                if verbose:
                    print(f"Error for {image_id}: {e}")
        
        return {
            "model_type": model_type,
            "model_name": MODEL_CONFIGS[model_type]["name"],
            "device": device,
            "success": True,
            "num_generated": len([r for r in model_results if r["success"]]),
            "results": model_results
        }
        
    except Exception as e:
        return {
            "model_type": model_type,
            "model_name": MODEL_CONFIGS[model_type]["name"],
            "device": device,
            "success": False,
            "error": str(e),
            "results": []
        }


def generate_images_for_all_models(
    prompts: List[tuple],  # List of (image_id, caption) pairs
    output_dir: str = "generated_images",
    models: Optional[List[str]] = None,
    model_paths: Optional[Dict[str, str]] = None,
    config_paths: Optional[Dict[str, str]] = None,
    device: int = 0,
    batch_size: int = 1,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate images from caption prompts using all specified models sequentially.
    
    Args:
        prompts: List of (image_id, caption) tuples
        output_dir: Output directory for generated images
        models: List of model types to use (default: all models)
        model_paths: Custom model paths dict {model_type: path}
        config_paths: Custom config paths dict {model_type: path}
        device: GPU device ID to use (default: 0)
        batch_size: Number of prompts to process in each batch (default: 1)
        seed: Random seed for reproducibility
        verbose: Print progress messages
    
    Returns:
        Dictionary with generation results and metadata
    """
    # Use all models if not specified
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"Generating images for {len(prompts)} prompts...")
        print(f"Using {len(models)} models: {', '.join(models)}")
        print(f"Device: GPU {device}")
        print(f"Batch size: {batch_size}")
        print(f"Output directory: {output_dir}")
    
    # Prepare model configurations
    model_configs = []
    for model_type in models:
        model_config = MODEL_CONFIGS[model_type]
        
        # Get model path
        if model_paths and model_type in model_paths:
            model_path = model_paths[model_type]
        else:
            model_path = model_config["default_path"]
        
        # Get config path if needed
        config_path = None
        if model_config.get("requires_config"):
            if config_paths and model_type in config_paths:
                config_path = config_paths[model_type]
            else:
                config_path = model_config.get("default_config")
        
        # Create model-specific output directory
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(exist_ok=True, parents=True)
        
        model_configs.append({
            "model_type": model_type,
            "model_path": model_path,
            "config_path": config_path,
            "output_dir": model_output_dir,
            "device": device
        })
    
    # Generate images sequentially
    all_results = []
    
    if verbose:
        print(f"\n⏩ Running {len(models)} models sequentially...")
    
    for cfg in model_configs:
        result = _generate_for_single_model(
            model_type=cfg["model_type"],
            model_path=cfg["model_path"],
            config_path=cfg["config_path"],
            prompts=prompts,
            output_dir=cfg["output_dir"],
            device=cfg["device"],
            seed=seed,
            batch_size=batch_size,
            verbose=verbose
        )
        all_results.append(result)
    
    # Create summary
    summary = {
        "num_prompts": len(prompts),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        "num_models": len(all_results),
        "successful_models": len([r for r in all_results if r["success"]]),
        "results": all_results,
    }
    
    # Save metadata
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {summary['successful_models']}/{summary['num_models']} models succeeded")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
    
    return summary


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate images from prompts using supported models"
    )
    
    parser.add_argument("--prompts_json", type=str, required=True,
                       help="JSON file with list of [image_id, caption] pairs")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                       help="Output directory (default: generated_images)")
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
    
    # Load prompts from JSON
    with open(args.prompts_json, 'r') as f:
        prompts = json.load(f)
    
    if args.devices and len(args.devices) > 1 and not args.quiet:
        print("Warning: multiple devices were provided; using the first one for sequential execution.")

    if not args.no_multiprocessing and not args.quiet:
        print("Note: multiprocessing is currently not implemented in this script; running sequentially.")

    selected_device = args.devices[0] if args.devices else 0

    # Call the main function
    generate_images_for_all_models(
        prompts=prompts,
        output_dir=args.output_dir,
        models=args.models,
        device=selected_device,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
