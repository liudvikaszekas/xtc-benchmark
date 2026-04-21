#!/usr/bin/env python3
"""
VLLM Inference Only - Minimal Script for Container Execution

This script contains ONLY VLLM inference logic with minimal dependencies.
It's designed to run in a clean VLLM container without pandas, gas, or other dependencies.

Input: Pickle file with preprocessed object data
Output: Pickle file with LLM-generated descriptions
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:
    print("ERROR: VLLM library not found")
    sys.exit(1)


def initialize_vllm(model_name: str, num_gpus: int, max_tokens: int, temperature: float):
    """Initialize VLLM model."""
    print(f"Initializing VLLM model: {model_name} on {num_gpus} GPUs...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 10},
        max_num_seqs=5,
        dtype="auto",
    )
    print("✓ VLLM model initialized")
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</s>", "\n\n", ".", "!"]
    )
    
    return llm, sampling_params


def format_attributes(attributes: Dict[str, List[str]]) -> str:
    """Format attributes into a readable string."""
    if not attributes:
        return "no specific attributes provided"
    
    attr_parts = []
    for attr_type, values in attributes.items():
        if values:
            values_str = ", ".join(values)
            attr_parts.append(f"{attr_type}: {values_str}")
    
    return "; ".join(attr_parts) if attr_parts else "no specific attributes provided"


def create_prompt(label: str, attributes: Dict[str, List[str]]) -> str:
    """Create prompt for object description generation."""
    attr_str = format_attributes(attributes)
    
    placeholder = "<|image_pad|>"
    
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant that generates natural language descriptions.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"The image shows a/an {label} object. "
        f"Provided attributes: {attr_str}. "
        f"Generate a concise, natural description (1-2 sentences max) that incorporates ALL the provided attributes. "
        f"IMPORTANT: Describe ONLY the visual content. Do NOT mention image artifacts like 'cropped', 'pixellated', 'low resolution', or 'background'. "
        f"Start with lowercase 'a' or 'an' (NOT capitalized). Do NOT end with punctuation. "
        f"Example format: 'a large black TV with red reflections'<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt


def batch_generate_descriptions(
    llm: LLM,
    sampling_params: SamplingParams,
    batch_data: List[Dict[str, Any]],
    batch_size: int
) -> List[str]:
    """
    Generate descriptions for a batch of objects.
    
    Args:
        llm: VLLM model instance
        sampling_params: Sampling parameters
        batch_data: List of dicts with 'label', 'attributes', 'object_image' keys
        batch_size: Batch size for inference
        
    Returns:
        List of description strings
    """
    all_descriptions = []
    total_batches = (len(batch_data) + batch_size - 1) // batch_size
    
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} objects)...")
        
        # Prepare batch inputs
        batch_inputs = []
        for data in batch:
            prompt = create_prompt(data['label'], data['attributes'])
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": data['object_image']}
            })
        
        # Batch generation with VLLM
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        
        # Extract descriptions
        for output in outputs:
            description = output.outputs[0].text.strip()
            description = description.rstrip('.!?,;:')
            all_descriptions.append(description)
    
    return all_descriptions


def main():
    parser = argparse.ArgumentParser(description="VLLM inference only (container-safe)")
    parser.add_argument("--input-pickle", required=True, help="Input pickle file with preprocessed data")
    parser.add_argument("--output-pickle", required=True, help="Output pickle file for results")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct", help="VLLM model name")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature")
    
    args = parser.parse_args()
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.input_pickle}...")
    with open(args.input_pickle, 'rb') as f:
        data = pickle.load(f)
    
    batch_data = data['batch_data']
    indices = data['indices']
    metadata = data.get('metadata', {})
    
    print(f"Loaded {len(batch_data)} objects to process")
    
    # Initialize VLLM
    llm, sampling_params = initialize_vllm(
        args.model,
        args.num_gpus,
        args.max_tokens,
        args.temperature
    )
    
    # Generate descriptions
    descriptions = batch_generate_descriptions(
        llm,
        sampling_params,
        batch_data,
        args.batch_size
    )
    
    # Save results
    print(f"Saving results to {args.output_pickle}...")
    results = {
        'descriptions': descriptions,
        'indices': indices,
        'metadata': metadata
    }
    
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Generated {len(descriptions)} descriptions")
    print("✓ VLLM inference complete")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
