#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Union, Tuple


def rgb2id(color: Union[np.ndarray, Tuple[int, int, int]]):
    """Converts a given color to the internal segmentation id
    Adapted from https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L73
    """
    if isinstance(color, np.ndarray):
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map: Union[np.ndarray, int]):
    """Converts segmentation id to RGB color"""
    if isinstance(id_map, np.ndarray):
        assert id_map.max() <= 256 * 256 * 256 - 1, id_map.max()
        rgb_map = np.empty((*id_map.shape, 3), dtype=np.uint8)
        rgb_map[..., 0] = id_map % 256
        rgb_map[..., 1] = (id_map // 256) % 256
        rgb_map[..., 2] = (id_map // 256 // 256) % 256
        return rgb_map
    color = [id_map % 256, (id_map // 256) % 256, (id_map // 256 // 256) % 256]
    return color


def merge_mask(seg_path, id_mapping, output_path):
    """Apply id mapping to a segmentation mask and save the result"""
    seg_img = np.array(Image.open(seg_path))
    id_map = rgb2id(seg_img)
    
    # Apply the id mapping
    merged_id_map = id_map.copy()
    for old_id, new_id in id_mapping.items():
        merged_id_map[id_map == int(old_id)] = int(new_id)
    
    # Convert back to RGB
    merged_rgb = id2rgb(merged_id_map)
    Image.fromarray(merged_rgb).save(output_path)
    print(f"Saved merged mask to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Apply ID merging to segmentation masks')
    parser.add_argument('--anno', required=True, help='Path to anno.json file')
    parser.add_argument('--merged', required=True, help='Path to anno_merged.json file')
    parser.add_argument('--output-dir', required=True, help='Directory to save merged masks')
    args = parser.parse_args()
    
    # Load annotation files
    with open(args.anno, 'r') as f:
        anno_data = json.load(f)
    
    with open(args.merged, 'r') as f:
        merged_mappings = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base directory for resolving relative paths
    anno_dir = Path(args.anno).parent
    
    # Process each image
    for image_data in anno_data['data']:
        image_id = str(image_data['image_id'])
        seg_file = image_data['pan_seg_file_name']
        
        # Resolve the segmentation file path
        seg_path = anno_dir / seg_file
        
        if not seg_path.exists():
            print(f"Warning: Segmentation file not found: {seg_path}")
            continue
        
        # Get the id mapping for this image
        id_mapping = merged_mappings.get(image_id, {})
        
        # Create output path
        output_path = Path(args.output_dir) / seg_file
        
        # Merge and save
        merge_mask(seg_path, id_mapping, output_path)
    
    print(f"\nAll merged masks saved to {args.output_dir}")


if __name__ == '__main__':
    main()

