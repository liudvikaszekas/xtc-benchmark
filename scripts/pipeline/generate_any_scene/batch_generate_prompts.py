#!/usr/bin/env python3
"""
Batch Generate Prompts from Scene Graphs

This script iterates over a directory of scene graph JSON files, 
generates captions using the PSGEval method, and saves the results.
"""

import json
import argparse
import sys
from pathlib import Path

# Add current directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

try:
    from generate_caption_from_sg import generate_caption_from_scene_graph
except ImportError:
    # If running from outside, try to handle relative imports or check python path
    try:
        from .generate_caption_from_sg import generate_caption_from_scene_graph
    except ImportError:
        print("Error: Could not import generate_caption_from_scene_graph. Make sure you are in the generate_any_scene directory or it is in your PYTHONPATH.")
        sys.exit(1)


def process_scene_graphs(scene_graphs_dir, output_dir):
    input_path = Path(scene_graphs_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return False

    output_path.mkdir(parents=True, exist_ok=True)
    
    scene_graph_files = sorted(list(input_path.glob("scene-graph_*.json")))
    if not scene_graph_files:
        print(f"Warning: No scene graph files found in {input_path}")
        return False
        
    print(f"Found {len(scene_graph_files)} scene graph files in {input_path}")
    
    prompts = []
    successful = 0
    failed = 0
    
    for sg_file in scene_graph_files:
        try:
            with open(sg_file, 'r') as f:
                scene_graph_data = json.load(f)
            
            # Simple format check/conversion if needed (logic moved from workflow step)
            if 'boxes' not in scene_graph_data:
                print(f"Skipping {sg_file.name}: 'boxes' key missing")
                failed += 1
                continue
                
            image_id = scene_graph_data.get('image_id')
            
            # Generate caption
            caption, sg_json = generate_caption_from_scene_graph(scene_graph_data)
            
            prompt_entry = {
                "image_id": image_id,
                "file_name": scene_graph_data.get('file_name', f"image_{image_id}.jpg"),
                "prompt": caption,
                "scene_graph": sg_json,
                "source_file": sg_file.name
            }
            
            prompts.append(prompt_entry)
            successful += 1
            
            if successful % 10 == 0:
                print(f"Processed {successful}/{len(scene_graph_files)} scene graphs...")
                
        except Exception as e:
            print(f"Error processing {sg_file.name}: {e}")
            failed += 1
            continue
            
    # Save structured JSON
    output_json = output_path / 'prompts.json'
    with open(output_json, 'w') as f:
        json.dump(prompts, f, indent=2)
        
    # Save simple text file
    output_txt = output_path / 'prompts.txt'
    with open(output_txt, 'w') as f:
        for p in prompts:
            f.write(f"Image {p['image_id']}: {p['prompt']}\n\n")
            
    print(f"\n✓ Generated {successful} prompts")
    print(f"✓ Saved to: {output_json}")
    print(f"✓ Saved text version to: {output_txt}")
    
    if failed > 0:
        print(f"⚠ Failed to process {failed} scene graphs")
        
    return successful > 0


def main():
    parser = argparse.ArgumentParser(description="Generate prompts from scene graphs")
    parser.add_argument("--input-dir", required=True, help="Directory containing scene graph JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated prompts")
    
    args = parser.parse_args()
    
    success = process_scene_graphs(args.input_dir, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
