#!/usr/bin/env python3
"""
Refine VQA Questions for Natural Flow

Takes generated VQA questions and refines them into more natural, flowing questions using an LLM.
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm

class QuestionRefiner:
    """Refines questions for better natural language flow."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        num_gpus: int = 2,
        max_tokens: int = 128, # Questions are short
        temperature: float = 0.4
    ):
        """
        Initialize the refiner with an LLM client.
        """
        print(f"Initializing LLM for question refinement: {model_name}")
        
        try:
            from vllm import LLM
            from vllm.sampling_params import SamplingParams
        except ImportError:
            raise RuntimeError("VLLM not installed. Run: pip install vllm")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            import torch
            available_gpus = torch.cuda.device_count()
        except Exception as e:
            raise RuntimeError(f"Could not detect GPUs: {e}")

        if available_gpus < num_gpus:
            print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
            num_gpus = available_gpus
            
        if num_gpus == 0:
             raise RuntimeError("No GPUs available.")

        # Initialize VLLM for text-only generation
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="auto",
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>", "<|im_end|>", "\n\n"] 
        )
        
        print("✓ LLM initialized for question refinement")
    
    def _create_refinement_prompt(self, original_question: str) -> str:
        """
        Create a prompt for the LLM to refine question structure.
        """
        # Qwen chat format
        system_message = (
            "You are an expert at asking natural questions about images. "
            "Your task is to take a generated question (which might be stilted, repetitive, or formulaic) "
            "and rewrite it to sound more natural, as a human would ask it. "
            "CRITICAL: Preserve the exact meaning and key constraints of the original question. "
            "Do not change the intent or the answer."
        )
        
        user_message = (
            f"Rewrite the following question to be more natural and flowing.\n"
            f"Original Question: {original_question}\n"
            f"Refined Question:"
        )
        
        prompt = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        return prompt
    
    def refine_batch(self, questions: List[str]) -> List[str]:
        """
        Refine multiple questions in batch.
        """
        llm_prompts = [self._create_refinement_prompt(q) for q in questions]
        
        outputs = self.llm.generate(llm_prompts, self.sampling_params)
        
        refined_questions = []
        for i, output in enumerate(outputs):
            if output.outputs and len(output.outputs) > 0:
                refined = output.outputs[0].text.strip()
                # Remove quotes if model adds them
                if refined.startswith('"') and refined.endswith('"'):
                    refined = refined[1:-1]
                refined_questions.append(refined)
            else:
                refined_questions.append(questions[i]) # Fallback
        
        return refined_questions


def parse_args():
    parser = argparse.ArgumentParser(description="Refine VQA questions using LLM")
    parser.add_argument("--input-file", required=True, help="Input questions JSON file")
    parser.add_argument("--output-file", required=True, help="Output questions JSON file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct", help="Model to use")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (logic only, VLLM handles batching)")
    parser.add_argument("--debug", action="store_true", help="Run on small subset")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)
        
    print(f"Loading questions from {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    questions_list = data["questions"]
    if args.debug:
        questions_list = questions_list[:20]
        print("DEBUG: Processing only first 20 questions")
        
    # Extract question strings
    raw_questions = [q["question"] for q in questions_list]
    
    # Initialize refiner
    try:
        refiner = QuestionRefiner(
            model_name=args.model,
            num_gpus=args.num_gpus
        )
    except Exception as e:
        print(f"Failed to initialize refiner: {e}")
        sys.exit(1)
        
    # Process in batches
    refined_texts = []
    batch_size = 512 # Send 512 prompts to VLLM at once
    
    print(f"Refining {len(raw_questions)} questions...")
    for i in tqdm(range(0, len(raw_questions), batch_size)):
        batch = raw_questions[i:i+batch_size]
        refined_batch = refiner.refine_batch(batch)
        refined_texts.extend(refined_batch)
        
    # Update data structure
    for q, refined in zip(questions_list, refined_texts):
        q["original_question"] = q["question"]
        q["question"] = refined
        q["is_refined"] = True
        
    # Save output
    print(f"Saving refined questions to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({"questions": questions_list}, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()
