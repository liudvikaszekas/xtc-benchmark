#!/usr/bin/env python3
"""
Generate Prompts from Scene Graphs (Pipeline Step 6)

This script:
1. Loads merged scene graphs (from Step 4: Graph Merging)
2. Loads attributes (from Step 5: Attribute Generation)
3. Merges attributes into scene graphs
4. Generates natural language prompts
5. Saves prompts.json

Usage:
    python generate_prompts.py --run_dir <run_dir>
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from utils_group_prompts import process_scene_graph_for_prompts
except ImportError:
    print(f"Error: Could not import utils_group_prompts")
    sys.exit(1)

import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Pipeline run directory")
    parser.add_argument("--refine-sentences", action="store_true", help="Refine prompt sentences for natural flow")
    parser.add_argument("--refine-objects", action="store_true", help="Refine object descriptions using LLM")
    parser.add_argument("--img-dir", type=str, help="Directory containing original images (required for object refinement)")
    parser.add_argument("--seg-dir", type=str, help="Directory containing segmentation masks (optional, for object refinement)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="LLM model name for refinement")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use for refinement")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for refinement")
    parser.add_argument("--max-tokens", type=int, default=25000, help="Max tokens for refinement")
    parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for refinement")
    parser.add_argument("--llm-env", type=str, default=None, help="Conda environment for LLM refinement")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    # We still need root_dir for some later script calls (if not yet moved)
    # We'll set root_dir based on the assumption that
    # the script is at benchmark/scripts/pipeline/generate_prompts.py
    curr_dir = Path(__file__).resolve().parent
    root_dir = curr_dir.parents[2]
    
    sg_dir = run_dir / "4_graph_merge_gt"
    attr_dir = run_dir / "5_attributes_gt"
    out_dir = run_dir / "6_prompt_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating prompts in {out_dir}")
    out_file = out_dir / "prompts.json"

    print(f"Reading Scene Graphs from {sg_dir}")
    
    attributes_data = []
    attr_file = attr_dir / "attributes.json"
    if attr_file.exists():
        print(f"Reading Attributes from {attr_file}")
        with open(attr_file) as f:
            attributes_data = json.load(f)
    else:
        print(f"Warning: attributes.json not found in {attr_dir}. Prompts will lack attributes.")

    # Index attributes by (image_id, seg_id)
    attr_map = {}
    for item in attributes_data:
        # item: {image_id, seg_id, attributes: {...}}
        attr_map[(item['image_id'], item['seg_id'])] = item.get('attributes', {})
        
    sg_files = list(sg_dir.glob("scene-graph_*.json"))
    print(f"Found {len(sg_files)} scene graphs.")
    
    # We will save merged scene graphs to a temporary or dedicated directory
    # so that object refinement can read them.
    merged_sg_dir = run_dir / "sg_with_attributes"
    merged_sg_dir.mkdir(parents=True, exist_ok=True)
    
    prompts_output = []
    
    for sg_file in sg_files:
        try:
            with open(sg_file) as f:
                sg = json.load(f)
        except Exception as e:
            print(f"Error loading {sg_file}: {e}")
            continue
            
        image_id = sg.get('image_id')
        if image_id is None:
            continue
        
        # Merge Attributes into Boxes
        for box in sg.get("boxes", []):
            seg_ids = box.get("seg_ids", [])
            
            # If it's a group (multiple seg_ids), we want to attach attributes to each member.
            if len(seg_ids) > 1:
                 members = []
                 for sid in seg_ids:
                     attrs = attr_map.get((image_id, sid), {})
                     members.append({"seg_id": sid, "attributes": attrs})
                 box["member_attributes"] = members
            elif len(seg_ids) == 1:
                # Single object
                sid = seg_ids[0]
                attrs = attr_map.get((image_id, sid), {})
                box["attributes"] = attrs
            elif box.get('id') is not None: 
                # fallback if seg_ids missing but id present (e.g. from simpler pipeline)
                sid = box['id']
                attrs = attr_map.get((image_id, sid), {})
                box["attributes"] = attrs
        
        # Save the merged scene graph
        with open(merged_sg_dir / sg_file.name, 'w') as f:
            json.dump(sg, f, indent=2)
            
        # If not refining objects, generate Prompt using imported utility
        if not args.refine_objects:
            try:
                # We want to enable relationship descriptions
                res = process_scene_graph_for_prompts(sg, include_relationships=True)
                prompts_output.append(res)
            except Exception as e:
                print(f"Error generating prompt for image {image_id}: {e}")
            
    if args.refine_objects:
        print("\n=== Applying Per-Object LLM Refinement ===")
        if not args.img_dir:
            print("Error: --img-dir is required when --refine-objects is True")
            sys.exit(1)
            
        refine_script = curr_dir / "generate_any_scene" / "batch_generate_prompts_llm.py"
        if not refine_script.exists():
            print(f"Error: Object refinement script not found at {refine_script}")
            sys.exit(1)
            
        cmd = ["python", str(refine_script),
               "--input-dir", str(merged_sg_dir),
               "--image-dir", args.img_dir,
               "--output-dir", str(out_dir),
               "--model", args.model,
               "--num-gpus", str(args.num_gpus),
               "--batch-size", str(args.batch_size),
               "--max-tokens", str(args.max_tokens),
               "--temperature", str(args.temperature)]
               
        if args.seg_dir:
             cmd.extend(["--seg-dir", args.seg_dir])
             
        if args.llm_env:
            cmd = ["conda", "run", "-n", args.llm_env, "--no-capture-output"] + cmd
            
        print(f"Executing object refinement: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("✓ Successfully generated prompts with object refinement")
        except subprocess.CalledProcessError as e:
            print(f"Error: Object refinement script failed with return code {e.returncode}")
            sys.exit(1)
    else:
        # Save prompts.json generated internally
        with open(out_file, 'w') as f:
            json.dump(prompts_output, f, indent=2)
            
        print(f"Saved {len(prompts_output)} prompts to {out_file}")

    # --- Sentence Refinement ---
    if args.refine_sentences:
        print("\n=== Applying Sentence-level Refinement ===")
        refine_script = curr_dir / "generate_any_scene" / "refine_prompt_sentences.py"
        if not refine_script.exists():
            print(f"Error: Refinement script not found at {refine_script}")
            sys.exit(1)
            
        refined_file = out_dir / "prompts_refined.json"
        
        # Base command
        cmd = ["python", str(refine_script), 
               "--prompts-file", str(out_file), 
               "--output-file", str(refined_file),
               "--model", args.model,
               "--tensor-parallel-size", str(args.num_gpus),
               "--batch-size", str(args.batch_size),
               "--max-tokens", str(args.max_tokens),
               "--temperature", str(args.temperature)]
               
        # Wrap with conda run if env specified
        if args.llm_env:
            cmd = ["conda", "run", "-n", args.llm_env, "--no-capture-output"] + cmd
            
        print(f"Executing refinement: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            if refined_file.exists():
                # Replace original with refined
                import shutil
                shutil.copy2(out_file, out_dir / "prompts_unrefined.json")
                shutil.move(refined_file, out_file)
                print("✓ Successfully refined prompts and updated prompts.json")
            else:
                print("Error: Refined file was not created.")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error: Refinement script failed with return code {e.returncode}")
            sys.exit(1)

    return 0

if __name__ == "__main__":
    exit(main())
