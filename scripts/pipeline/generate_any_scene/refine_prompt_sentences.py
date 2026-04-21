#!/usr/bin/env python3
"""
Refine Prompt Sentences for Natural Flow

Takes prompts with object descriptions and relationships and refines them
into more natural, flowing descriptions using an LLM.

This script improves the sentence structure to better integrate:
- Object descriptions (potentially enhanced by VLM)
- Spatial relationships
- Overall scene coherence
"""

import json
import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional


class PromptSentenceRefiner:
    """Refines prompt sentences for better natural language flow."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        num_gpus: int = 4,
        max_tokens: int = 300,
        temperature: float = 0.4
    ):
        """
        Initialize the refiner with an LLM client.
        
        Args:
            model_name: LLM model name
            num_gpus: Number of GPUs to use
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
        """
        print(f"Initializing LLM for sentence refinement: {model_name}")
        
        # Import VLLM here to avoid import errors if not needed
        try:
            from vllm import LLM
            from vllm.sampling_params import SamplingParams
        except ImportError:
            raise RuntimeError("VLLM not installed. Run: pip install vllm")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Detect available GPUs and fail early if insufficient
        try:
            import torch
            available_gpus = torch.cuda.device_count()
        except Exception as e:
            raise RuntimeError(
                "Could not detect GPUs (torch not available). "
                f"Requested {num_gpus} GPU(s). Ensure CUDA is visible and torch is installed. Original error: {e}"
            )

        if available_gpus <= 0:
            raise RuntimeError("No GPUs available for sentence refinement (CUDA not visible).")

        if available_gpus < num_gpus:
            raise RuntimeError(
                f"Requested {num_gpus} GPU(s) but only {available_gpus} detected. "
                "Please reduce --num-gpus or run on a node with more GPUs."
            )

        # Initialize VLLM for text-only generation
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95,
            max_model_len=32768,
            dtype="auto",
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>", "<|im_end|>"]
        )
        
        print("✓ LLM initialized for sentence refinement")
    
    def _create_refinement_prompt(
        self, 
        original_prompt: str, 
        object_descriptions: Optional[Dict[str, str]] = None,
        relation_descriptions: Optional[List[str]] = None
    ) -> str:
        """
        Create a prompt for the LLM to refine sentence structure.
        
        Args:
            original_prompt: Original prompt text to refine
            object_descriptions: Optional dictionary of object descriptions
            relation_descriptions: Optional list of relation descriptions
            
        Returns:
            Formatted prompt for the LLM
        """
        # Qwen chat format
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
    
    def refine_prompt(
        self, 
        original_prompt: str,
        object_descriptions: Optional[Dict[str, str]] = None,
        relation_descriptions: Optional[List[str]] = None
    ) -> str:
        """
        Refine a single prompt for better sentence flow.
        
        Args:
            original_prompt: Original prompt text
            object_descriptions: Optional dictionary of object descriptions
            relation_descriptions: Optional list of relation descriptions
            
        Returns:
            Refined prompt text
        """
        # Create refinement prompt
        prompt = self._create_refinement_prompt(original_prompt, object_descriptions, relation_descriptions)
        
        # Generate refined version
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        if outputs and len(outputs) > 0:
            refined = outputs[0].outputs[0].text.strip()
            return refined
        
        # Fallback to original if generation fails
        print("Warning: Refinement failed, using original prompt")
        return original_prompt
    
    def refine_batch(
        self, 
        prompts: List[str],
        batch_object_descriptions: Optional[List[Dict[str, str]]] = None,
        batch_relation_descriptions: Optional[List[List[str]]] = None
    ) -> List[str]:
        """
        Refine multiple prompts in batch.
        
        Args:
            prompts: List of original prompts
            batch_object_descriptions: Optional list of dictionaries of object descriptions
            batch_relation_descriptions: Optional list of lists of relation descriptions
            
        Returns:
            List of refined prompts
        """
        # Create all refinement prompts
        llm_prompts = []
        for i, p in enumerate(prompts):
            obj_descs = batch_object_descriptions[i] if batch_object_descriptions else None
            rel_descs = batch_relation_descriptions[i] if batch_relation_descriptions else None
            llm_prompts.append(self._create_refinement_prompt(p, obj_descs, rel_descs))
        
        # Generate refined versions in batch
        outputs = self.llm.generate(llm_prompts, self.sampling_params)
        
        refined_prompts = []
        for i, output in enumerate(outputs):
            if output.outputs and len(output.outputs) > 0:
                refined = output.outputs[0].text.strip()
                refined_prompts.append(refined)
            else:
                print(f"Warning: Refinement failed for prompt {i}, using original")
                refined_prompts.append(prompts[i])
        
        return refined_prompts


def _wait_for_slurm_job(job_id: str, poll_interval: int = 5, log_dir: Optional[Path] = None, prefix: str = "prompt-refine") -> None:
    """Poll Slurm until job completes and stream logs."""
    import time
    log_offsets = {}
    last_status = None
    
    while True:
        # Check job status
        result = subprocess.run(['squeue', '-j', job_id, '-h', '-o', '%T'], capture_output=True, text=True)
        status = result.stdout.strip()
        
        # Tail logs
        if log_dir:
            for log_file in log_dir.glob(f"*{job_id}.*"):
                if log_file.suffix not in ('.out', '.err'):
                    continue
                path_str = str(log_file)
                if path_str not in log_offsets:
                    log_offsets[path_str] = 0
                try:
                    with open(log_file, 'rb') as f:
                        f.seek(log_offsets[path_str])
                        chunk = f.read()
                        if chunk:
                            text = chunk.decode('utf-8', errors='replace')
                            lines = text.splitlines(keepends=True)
                            for line in lines:
                                print(f"[{prefix}] {line}", end='')
                            log_offsets[path_str] = f.tell()
                except Exception:
                    pass

        if not status:
            # Job finished
            break
            
        if status != last_status:
            print(f"\n[{prefix}] [Job {job_id} status: {status}]")
            last_status = status
            
        time.sleep(poll_interval)
    
    print(f"\n[{prefix}] Job finished processing.")


def _generate_refinement_slurm_script(
    input_pickle: Path,
    output_pickle: Path,
    slurm_config: Dict,
    llm_config: Dict,
    log_dir: Optional[Path] = None
) -> str:
    """Generate Slurm batch script for text refinement."""
    account = slurm_config.get('account', 'sci-zacharatou')
    partition = slurm_config.get('partition', 'gpu-batch')
    constraint = slurm_config.get('constraint')
    gpus_type = slurm_config.get('gpus_type', 'rtx_pro_6000')
    num_gpus = llm_config.get('num_gpus', 4)
    mem = slurm_config.get('mem', '100G')
    cpus = slurm_config.get('cpus', 8)
    time_limit = slurm_config.get('time', '02:00:00')
    
    model = llm_config.get('model', 'Qwen/Qwen2.5-VL-7B-Instruct')
    batch_size = llm_config.get('batch_size', 8)
    max_tokens = llm_config.get('max_tokens', 300)
    temperature = llm_config.get('temperature', 0.4)
    
    use_container = slurm_config.get('use_container', True)
    container_name = slurm_config.get('container_name', 'vllm')
    container_mounts = slurm_config.get('container_mounts', '/sc/home:/sc/home')
    
    # Get script path
    script_dir = Path(__file__).parent
    vllm_script = script_dir / 'vllm_text_refinement.py'
    
    log_path = log_dir if log_dir else output_pickle.parent
    if log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    
    script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name=vllm-refine-text
#SBATCH --partition={partition}
#SBATCH --gpus={gpus_type}:{num_gpus}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --output={log_path}/slurm_%j.out
#SBATCH --error={log_path}/slurm_%j.err
"""
    if constraint:
        script += f"#SBATCH --constraint={constraint}\n"
    
    if use_container:
        script += f"""#SBATCH --container-name={container_name}
#SBATCH --container-mounts={container_mounts}

srun python3 {vllm_script} \\
    --input-pickle {input_pickle} \\
    --output-pickle {output_pickle} \\
    --model {model} \\
    --num-gpus {num_gpus} \\
    --batch-size {batch_size} \\
    --max-tokens {max_tokens} \\
    --temperature {temperature}
"""
    else:
        script += f"""
srun python3 {vllm_script} \\
    --input-pickle {input_pickle} \\
    --output-pickle {output_pickle} \\
    --model {model} \\
    --num-gpus {num_gpus} \\
    --batch-size {batch_size} \\
    --max-tokens {max_tokens} \\
    --temperature {temperature}
"""
    
    return script


def submit_refinement_slurm_job(
    prompts: List[str],
    slurm_config: Dict,
    llm_config: Dict,
    log_dir: Optional[Path] = None,
    object_descriptions: Optional[List[Dict[str, str]]] = None,
    relation_descriptions: Optional[List[List[str]]] = None
) -> List[str]:
    """Submit Slurm job for text refinement using the proven pattern."""
    import pickle
    
    # Use log_dir or current directory
    if log_dir is None:
        log_dir = Path.cwd() / 'logs'
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory in project for pickle files
    temp_dir = log_dir / 'tmp_refinement'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    input_pickle = temp_dir / f"refine_input_{timestamp}.pkl"
    output_pickle = temp_dir / f"refine_output_{timestamp}.pkl"
    script_path = temp_dir / f"refine_job_{timestamp}.sh"
    
    try:
        print(f"  Preparing Slurm job for {len(prompts)} prompts...")
        with open(input_pickle, 'wb') as f:
            pickle.dump({
                'prompts': prompts, 
                'object_descriptions': object_descriptions,
                'relation_descriptions': relation_descriptions,
                'metadata': {}
            }, f)
        
        # Generate Slurm script
        slurm_script = _generate_refinement_slurm_script(
            input_pickle, output_pickle, slurm_config, llm_config, log_dir
        )
        
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        
        # Submit job
        print(f"  Submitting Slurm job...")
        result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Slurm submission failed: {result.stderr}")
        
        # Parse job ID
        import re
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
        
        job_id = match.group(1)
        print(f"  ✓ Submitted job {job_id}, waiting for completion...")
        
        # Wait for completion and stream logs
        _wait_for_slurm_job(job_id, log_dir=log_dir)
        
        # Load results
        if not output_pickle.exists():
            raise RuntimeError(f"Output pickle not found: {output_pickle}")
        
        with open(output_pickle, 'rb') as f:
            results = pickle.load(f)
        
        refined_prompts = results['refined_prompts']
        print(f"  ✓ Received {len(refined_prompts)} refined prompts from Slurm job")
        
        return refined_prompts
        
    finally:
        # Clean up temporary files
        try:
            if input_pickle.exists():
                input_pickle.unlink()
            if output_pickle.exists():
                output_pickle.unlink()
            if script_path.exists():
                script_path.unlink()
            # Remove temp directory if empty
            if temp_dir.exists() and not list(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception as e:
            print(f"  Warning: Failed to clean up temp files: {e}")


def refine_prompts_file(
    prompts_file: Path,
    output_file: Path,
    llm_config: dict,
    batch_size: int = 8,
    max_prompts: Optional[int] = None,
    slurm_config: Optional[Dict] = None
) -> bool:
    """
    Refine prompts from a JSON file.
    
    Args:
        prompts_file: Input prompts.json file
        output_file: Output file for refined prompts
        llm_config: LLM configuration dict
        batch_size: Batch size for processing
        max_prompts: Optional limit on number of prompts to refine
        
    Returns:
        True if successful
    """
    if not prompts_file.exists():
        print(f"Error: Prompts file not found: {prompts_file}")
        return False
    
    # Load prompts
    print(f"Loading prompts from {prompts_file}")
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)
    
    print(f"Found {len(prompts_data)} prompts to refine")

    # Handle list format
    if isinstance(prompts_data, list):
        print("Converting list format to dictionary...")
        prompts_dict = {}
        for item in prompts_data:
            img_id = item.get('image_id')
            if img_id is not None:
                prompts_dict[str(img_id)] = item
        prompts_data = prompts_dict
    
    # Limit for debugging
    if max_prompts is not None and max_prompts > 0:
        original_count = len(prompts_data)
        prompts_data = dict(list(prompts_data.items())[:max_prompts])
        print(f"DEBUG: Limiting to {len(prompts_data)} prompts (out of {original_count})")
    
    # Extract prompts and descriptions in order
    image_ids = list(prompts_data.keys())
    prompts = [prompts_data[img_id]["prompt"] for img_id in image_ids]
    object_descriptions = [prompts_data[img_id].get("object_descriptions", {}) for img_id in image_ids]
    relation_descriptions = [prompts_data[img_id].get("relation_descriptions", []) for img_id in image_ids]
    
    # Refine using Slurm or local
    if slurm_config and slurm_config.get('use_slurm', False):
        print("Using Slurm mode for text refinement")
        refined_prompts = submit_refinement_slurm_job(
            prompts, 
            slurm_config, 
            llm_config, 
            log_dir=output_file.parent / 'logs',
            object_descriptions=object_descriptions,
            relation_descriptions=relation_descriptions
        )
    else:
        print("Using local mode for text refinement")
        # Initialize refiner
        refiner = PromptSentenceRefiner(
            model_name=llm_config.get("model", "Qwen/Qwen2.5-VL-32B-Instruct"),
            num_gpus=llm_config.get("num_gpus", 4),
            max_tokens=llm_config.get("max_tokens", 300),
            temperature=llm_config.get("temperature", 0.4)
        )
        
        # Process in batches
        refined_prompts = []
        for i in range(0, len(prompts), batch_size):
            batch_p = prompts[i:i+batch_size]
            batch_obj = object_descriptions[i:i+batch_size]
            batch_rel = relation_descriptions[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} ({len(batch_p)} prompts)")
            refined_batch = refiner.refine_batch(batch_p, batch_obj, batch_rel)
            refined_prompts.extend(refined_batch)
    
    # Update data with refined prompts
    refined_data = {}
    for img_id, refined_prompt in zip(image_ids, refined_prompts):
        refined_data[img_id] = prompts_data[img_id].copy()
        refined_data[img_id]["prompt"] = refined_prompt
        refined_data[img_id]["original_prompt"] = prompts_data[img_id]["prompt"]
    
    # Save refined prompts
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(refined_data, f, indent=2)
    
    print(f"✓ Refined {len(refined_data)} prompts")
    print(f"✓ Saved to {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Refine prompt sentences for more natural flow"
    )
    parser.add_argument(
        '--prompts-file',
        type=Path,
        required=True,
        help='Input prompts.json file'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        required=True,
        help='Output file for refined prompts'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help='LLM model name'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (legacy; same as --tensor-parallel-size)'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=4,
        help='Tensor parallel size / number of GPUs to use (preferred)'
    )
    parser.add_argument(
        '--slurm-config',
        type=Path,
        default=None,
        help='Optional Slurm config JSON to run refinement via Slurm'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=300,
        help='Maximum tokens for generation'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.4,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-prompts',
        type=int,
        default=None,
        help='Maximum number of prompts to refine (for debugging)'
    )
    
    args = parser.parse_args()
    
    # Prefer explicit tensor-parallel-size; fall back to num-gpus for compatibility
    effective_num_gpus = args.tensor_parallel_size
    if args.num_gpus is not None:
        effective_num_gpus = args.num_gpus

    llm_config = {
        "model": args.model,
        "num_gpus": effective_num_gpus,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature
    }

    # Load Slurm config if provided
    slurm_cfg = None
    if args.slurm_config:
        slurm_cfg_path = Path(args.slurm_config)
        if not slurm_cfg_path.exists():
            print(f"Error: Slurm config file not found: {slurm_cfg_path}")
            return 1
        with open(slurm_cfg_path, 'r') as f:
            slurm_cfg = json.load(f)

    success = refine_prompts_file(
        prompts_file=args.prompts_file,
        output_file=args.output_file,
        llm_config=llm_config,
        batch_size=args.batch_size,
        max_prompts=args.max_prompts,
        slurm_config=slurm_cfg
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
