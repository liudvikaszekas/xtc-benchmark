#!/usr/bin/env python3
"""
Convert scene graph descriptions to matching format.

This script converts scene graph JSON files into the format expected by match_openpsg_graphs.py.

Output format:
{
  "metadata": {...},
  "images": {
    "<image_id>": {
      "prediction": {
        "nodes": [{"id": "node_0", "label": "bed", "attributes": {...}}, ...],
        "edges": [{"source": "node_0", "target": "node_1", "relation": "on"}, ...]
      },
      "ground_truth": {
        "nodes": [...],
        "edges": [...]
      }
    }
  }
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def convert_sg_to_graph(scene_graph: Dict) -> Dict:
    """
    Convert a scene graph to the graph format expected by hungarian.py.
    
    Args:
        scene_graph: Scene graph with boxes and relations
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists
    """
    nodes = []
    edges = []

    # Track which IDs were actually added to the graph
    added_node_ids = set()
    box_to_node_ids = defaultdict(list)

    # Create nodes from boxes
    for box in scene_graph.get('boxes', []):
        label = box['label']
        box_idx = box.get('index')

        # If the node is merged and has member attributes, expand it
        if 'member_attributes' in box and box['member_attributes']:
            for member in box['member_attributes']:
                member_id = str(member.get('seg_id', member.get('id', f"node_{box['index']}_{len(added_node_ids)}")))
                node_data = {
                    'id': member_id,
                    'label': label,
                    'attributes': member.get('attributes', {}),
                    'index': box_idx
                }
                nodes.append(node_data)
                added_node_ids.add(member_id)
                if box_idx is not None:
                    box_to_node_ids[box_idx].append(member_id)
        else:
            # Standard node
            node_id = str(box.get('id', f"node_{box['index']}"))
            node_data = {
                'id': node_id,
                'label': label,
                'attributes': box.get('attributes', {}),
                'index': box_idx
            }
            nodes.append(node_data)
            added_node_ids.add(node_id)
            if box_idx is not None:
                box_to_node_ids[box_idx].append(node_id)

    # Create edges from relations
    for rel in scene_graph.get('relations', []):
        # Support both 'predicate' string and 'predicates' list
        predicate = rel.get('predicate')
        if not predicate and 'predicates' in rel and rel['predicates']:
            predicate = rel['predicates'][0]['predicate']

        if not predicate:
            continue

        # Map relations from box_index groups to all members
        subject_idx = rel.get('subject_index')
        object_idx = rel.get('object_index')

        if subject_idx is not None and object_idx is not None:
            source_ids = box_to_node_ids.get(subject_idx, [])
            target_ids = box_to_node_ids.get(object_idx, [])

            for source_id in source_ids:
                for target_id in target_ids:
                    if source_id in added_node_ids and target_id in added_node_ids:
                        edges.append({
                            'source': source_id,
                            'target': target_id,
                            'relation': predicate
                        })
        else:
            raise ValueError(f"Missing subject_index or object_index in relation: {rel}")
    
    return {
        'nodes': nodes,
        'edges': edges
    }


def extract_image_id_from_filename(filename: str) -> str:
    """
    Extract image ID from filename.
    Supports patterns like:
    - 'scene-graph-description_1_compact.json' -> '1'
    - 'scene-graph_1_compact.json' -> '1'
    - '1.png' -> '1'
    """
    # Try to extract from compact file pattern
    if '_compact.json' in filename:
        parts = filename.replace('_compact.json', '').split('_')
        # Handle scene-graph-description_1_compact.json
        if len(parts) >= 2 and parts[0] == 'scene-graph-description':
            return parts[1]
        # Handle scene-graph_1_compact.json
        if len(parts) >= 2 and parts[0] == 'scene-graph':
            return parts[1]
    
    # Try to extract numeric ID from filename
    match = re.search(r'(\d+)', filename)
    if match:
        return str(int(match.group(1)))
    
    # Fallback: use filename without extension
    return Path(filename).stem


def extract_id_from_file_path(file_path: str) -> str:
    """
    Extract matching ID from file path or filename.
    For paths like 'datasets/gt/1.png' or 'datasets/pt/1.png', extracts '1'.
    For filenames like '1.png', extracts '1'.
    """
    # Get the basename (filename with extension)
    basename = Path(file_path).name
    
    # Remove extension to get the base name
    stem = Path(basename).stem
    
    # Try to extract numeric ID
    match = re.search(r'(\d+)', stem)
    if match:
        return str(int(match.group(1)))
    
    # Fallback: return the stem (filename without extension)
    return stem


def load_scene_graphs_from_dir(directory: Path, match_by_filename: bool = False) -> Dict[str, Dict]:
    """
    Load all scene graph files from a directory and index by image_id.
    
    Args:
        directory: Directory containing scene graph JSON files
        match_by_filename: If True, extract matching ID from file_name field instead of image_id
    
    Returns:
        Dictionary mapping image_id (or filename-based ID) to scene graph data
    """
    # Support multiple naming patterns: scene-graph-description_*.json, scene-graph_*.json
    sg_files = list(directory.glob('scene-graph-description_*.json'))
    if not sg_files:
        sg_files = list(directory.glob('scene-graph_*.json'))
    
    graphs_by_id = {}
    for sg_file in sorted(sg_files):
        try:
            with open(sg_file, 'r') as f:
                scene_graph = json.load(f)
            
            if match_by_filename:
                # Extract ID from the file_name field (e.g., 'datasets/gt/1.png' -> '1')
                file_name = scene_graph.get('file_name', '')
                if file_name:
                    matching_id = extract_id_from_file_path(file_name)
                else:
                    # Fallback to filename extraction
                    matching_id = extract_image_id_from_filename(sg_file.name)
            else:
                # Use image_id from JSON or extract from filename
                matching_id = str(scene_graph.get('image_id'))
                if not matching_id or matching_id == 'None':
                    matching_id = extract_image_id_from_filename(sg_file.name)
            
            graphs_by_id[matching_id] = scene_graph
            print(f"  Loaded {sg_file.name} -> matching_id: {matching_id} (file_name: {scene_graph.get('file_name', 'N/A')})")
        except Exception as e:
            print(f"  Warning: Failed to load {sg_file.name}: {e}")
    
    return graphs_by_id


def main():
    parser = argparse.ArgumentParser(
        description='Convert scene graphs to matching format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Directory containing scene graph JSON files (legacy mode)'
    )
    parser.add_argument(
        '--gt-dir',
        type=str,
        default=None,
        help='Directory containing ground truth scene graph JSON files'
    )
    parser.add_argument(
        '--pt-dir',
        type=str,
        default=None,
        help='Directory containing predicted scene graph JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file for matching'
    )
    parser.add_argument(
        '--use-as-ground-truth',
        action='store_true',
        help='Use the same graphs as both prediction and ground truth (for testing, legacy mode)'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    # Determine mode: gt/pt matching vs legacy single directory
    if args.gt_dir and args.pt_dir:
        # New mode: match gt vs pt
        gt_dir = Path(args.gt_dir)
        pt_dir = Path(args.pt_dir)
        
        if not gt_dir.exists():
            print(f"Error: Ground truth directory not found: {gt_dir}")
            return 1
        if not pt_dir.exists():
            print(f"Error: Prediction directory not found: {pt_dir}")
            return 1
        
        print("Loading ground truth scene graphs...")
        gt_graphs = load_scene_graphs_from_dir(gt_dir, match_by_filename=True)
        print(f"Found {len(gt_graphs)} ground truth graphs")
        
        print("\nLoading predicted scene graphs...")
        pt_graphs = load_scene_graphs_from_dir(pt_dir, match_by_filename=True)
        print(f"Found {len(pt_graphs)} predicted graphs")
        
        # Find common image IDs
        common_ids = set(gt_graphs.keys()) & set(pt_graphs.keys())
        
        # If file_name-based matching fails, try matching by image_id field directly
        if not common_ids:
            print("No common IDs via file_name matching, trying image_id field...")
            gt_by_image_id = {}
            for key, sg in gt_graphs.items():
                iid = str(sg.get('image_id', key))
                gt_by_image_id[iid] = sg
            pt_by_image_id = {}
            for key, sg in pt_graphs.items():
                iid = str(sg.get('image_id', key))
                pt_by_image_id[iid] = sg
            common_ids = set(gt_by_image_id.keys()) & set(pt_by_image_id.keys())
            if common_ids:
                print(f"Found {len(common_ids)} matches via image_id field")
                gt_graphs = gt_by_image_id
                pt_graphs = pt_by_image_id
        
        print(f"\nFound {len(common_ids)} common image IDs to match")
        
        if not common_ids:
            gt_ids = sorted(set(gt_graphs.keys()))[:10]
            pt_ids = sorted(set(pt_graphs.keys()))[:10]
            print("ERROR: No common image IDs found between GT and PT directories.")
            print(f"  GT sample IDs: {gt_ids}{'...' if len(gt_graphs) > 10 else ''}")
            print(f"  PT sample IDs: {pt_ids}{'...' if len(pt_graphs) > 10 else ''}")
            print("  Ensure prediction images are named to match GT image IDs (e.g., COCO IDs).")
            return 1
        
        # Create output structure
        output_data = {
            'metadata': {
                'source': 'gt vs pt scene graphs',
                'gt_dir': str(gt_dir),
                'pt_dir': str(pt_dir),
                'num_images': len(common_ids),
                'format': 'hungarian_matching'
            },
            'images': {}
        }
        
        # Process each matched image ID
        for image_id in sorted(common_ids):
            gt_sg = gt_graphs[image_id]
            pt_sg = pt_graphs[image_id]
            
            gt_graph = convert_sg_to_graph(gt_sg)
            pt_graph = convert_sg_to_graph(pt_sg)
            
            output_data['images'][image_id] = {
                'ground_truth': gt_graph,
                'prediction': pt_graph
            }
            
            print(f"  Image {image_id}: GT({len(gt_graph['nodes'])} nodes, {len(gt_graph['edges'])} edges) "
                  f"vs PT({len(pt_graph['nodes'])} nodes, {len(pt_graph['edges'])} edges)")
        
    elif args.input_dir:
        # Legacy mode: single input directory
        input_dir = Path(args.input_dir)
        
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return 1
        
        # Find all scene graph files
        sg_files = list(input_dir.glob('scene-graph-description_*.json'))
        if not sg_files:
            sg_files = list(input_dir.glob('scene-graph_*.json'))
        
        if not sg_files:
            print(f"Error: No scene graph files found in {input_dir}")
            print("Expected pattern: scene-graph_*.json or scene-graph-description_*.json")
            return 1
        
        print(f"Found {len(sg_files)} scene graph files")
        
        # Create output structure
        output_data = {
            'metadata': {
                'source': 'scene graphs',
                'num_images': len(sg_files),
                'format': 'hungarian_matching'
            },
            'images': {}
        }
        
        # Process each file
        for sg_file in sorted(sg_files):
            print(f"Processing {sg_file.name}...")
            
            with open(sg_file, 'r') as f:
                scene_graph = json.load(f)
            
            image_id = str(scene_graph['image_id'])
            
            # Convert to graph format
            graph = convert_sg_to_graph(scene_graph)
            
            # Create image entry
            if args.use_as_ground_truth:
                # Use same graph as both prediction and ground truth
                output_data['images'][image_id] = {
                    'prediction': graph,
                    'ground_truth': graph
                }
            else:
                # Use as prediction only, create empty ground truth
                output_data['images'][image_id] = {
                    'prediction': graph,
                    'ground_truth': {
                        'nodes': [],
                        'edges': []
                    }
                }
            
            print(f"  Image {image_id}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    else:
        print("Error: Must specify either (--gt-dir and --pt-dir) or --input-dir")
        return 1
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Conversion complete")
    print(f"Output saved to: {output_path}")
    print(f"Total images: {len(output_data['images'])}")
    
    return 0


if __name__ == '__main__':
    exit(main())
