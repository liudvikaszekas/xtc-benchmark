#!/usr/bin/env python3
"""
VLLM Text Refinement Only - Minimal Script for Container Execution

This script contains ONLY VLLM inference logic for text refinement.
Designed to run in a clean VLLM container.

Input: Pickle file with prompt texts
Output: Pickle file with refined prompt texts
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:
    print("ERROR: VLLM library not found")
    sys.exit(1)


def initialize_vllm(model_name: str, num_gpus: int, max_tokens: int, temperature: float):
    """Initialize VLLM model for text-only refinement."""
    print(f"Initializing VLLM model: {model_name} on {num_gpus} GPUs...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        dtype="auto",
    )
    print("✓ VLLM model initialized")
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</s>", "<|im_end|>"]
    )
    
    return llm, sampling_params


def create_refinement_prompt(
    original_prompt: str,
    object_descriptions: dict = None,
    relation_descriptions: list = None
) -> str:
    """Create a prompt for the LLM to refine sentence structure."""
    system_message = (
        "You are an expert at writing natural, flowing scene descriptions. "
        "Your task is to take a structured scene description and rewrite it to sound more natural and cohesive. "
        "You MUST preserve ALL information provided and ensure every object and relationship mentioned is included in your description. "
        "CRITICAL: Do NOT add any objects, attributes, or relationships that are not present in the input. Use ONLY the provided information.\n"
        "CRITICAL: You MUST PRESERVE all object and group identifiers (e.g., PersonGroup_1, Object_2) exactly as they appear in parentheses or alongside objects. These identifiers are crucial for downstream tasks."
    )
    
    # Format structured data if provided
    structured_info = ""
    if object_descriptions:
        structured_info += "Object Descriptions:\n"
        for obj_id, desc in object_descriptions.items():
            structured_info += f"- {desc}\n"
    
    if relation_descriptions:
        structured_info += "\nRelationships:\n"
        for rel in relation_descriptions:
            structured_info += f"- {rel}\n"

    user_message = (
        f"Rewrite the following scene description to be more natural and flowing.\n\n"
    )
    
    if structured_info:
        user_message += f"Here is the detailed structured information that MUST all be included:\n{structured_info}\n"
    
    user_message += (
        f"Current draft description:\n{original_prompt}\n\n"
        f"Please integrate all details smoothly into a cohesive paragraph. "
        f"Preserve all specific attributes (colors, counts, positions) and relations. "
        f"IMPORTANT: Do NOT remove the object/group identifiers (like PersonGroup_1, Tree-merged_1, etc.). Keep them associated with their respective descriptions. "
        f"Output ONLY the refined description text.\n"
        f"Rewritten description:"
    )
    
    prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt


def batch_refine_prompts(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    object_descriptions: List[dict] = None,
    relation_descriptions: List[list] = None
) -> List[str]:
    """
    Refine a batch of prompts.
    
    Args:
        llm: VLLM model instance
        sampling_params: Sampling parameters
        prompts: List of original prompt texts
        object_descriptions: List of object description dicts
        relation_descriptions: List of relation description lists
        
    Returns:
        List of refined prompt strings
    """
    # Create refinement prompts
    llm_prompts = []
    for i in range(len(prompts)):
        obj_descs = object_descriptions[i] if object_descriptions else None
        rel_descs = relation_descriptions[i] if relation_descriptions else None
        llm_prompts.append(create_refinement_prompt(prompts[i], obj_descs, rel_descs))
    
    # Generate refined versions in batch
    print(f"Refining {len(prompts)} prompts...")
    outputs = llm.generate(llm_prompts, sampling_params)
    
    refined_prompts = []
    for i, output in enumerate(outputs):
        if output.outputs and len(output.outputs) > 0:
            refined = output.outputs[0].text.strip()
            refined_prompts.append(refined)
        else:
            print(f"Warning: Refinement failed for prompt {i}, using original")
            refined_prompts.append(prompts[i])
    
    return refined_prompts


def main():
    parser = argparse.ArgumentParser(description="VLLM-only text refinement")
    parser.add_argument('--input-pickle', type=Path, required=True)
    parser.add_argument('--output-pickle', type=Path, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num-gpus', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.4)
    
    args = parser.parse_args()
    
    # Load input
    print(f"Loading input from {args.input_pickle}")
    with open(args.input_pickle, 'rb') as f:
        data = pickle.load(f)
    
    prompts = data['prompts']
    print(f"Loaded {len(prompts)} prompts to refine")
    
    # Initialize VLLM
    llm, sampling_params = initialize_vllm(
        args.model, args.num_gpus, args.max_tokens, args.temperature
    )
    
    # Refine prompts
    refined_prompts = batch_refine_prompts(
        llm, 
        sampling_params, 
        prompts,
        data.get('object_descriptions'),
        data.get('relation_descriptions')
    )
    
    # Save output
    print(f"Saving output to {args.output_pickle}")
    with open(args.output_pickle, 'wb') as f:
        pickle.dump({
            'refined_prompts': refined_prompts,
            'metadata': data.get('metadata', {})
        }, f)
    
    print(f"✓ Refined {len(refined_prompts)} prompts successfully")


if __name__ == '__main__':
    main()
