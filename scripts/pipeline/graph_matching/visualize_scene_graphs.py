"""
Visualize scene graph predictions vs ground truth by overlaying segmentation masks on images.

This script loads the scene graphs JSON and PSG ground truth, then creates visualizations
showing the predicted and ground truth segmentation masks with labels overlaid on the original images.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys


def load_psg_data(psg_json_path: str) -> Dict:
    """Load PSG ground truth data."""
    print(f"Loading PSG data from {psg_json_path}...")
    with open(psg_json_path, 'r') as f:
        psg_data = json.load(f)
    
    # Create lookup by image_id
    by_image = {}
    for entry in psg_data['data']:
        image_id = entry['image_id']
        by_image[image_id] = entry
    
    return {
        'data': psg_data,
        'by_image': by_image,
        'thing_classes': psg_data['thing_classes'],
        'stuff_classes': psg_data['stuff_classes'],
        'predicate_classes': psg_data['predicate_classes']
    }


def load_scene_graphs(json_path: str) -> Dict:
    """Load scene graphs from JSON file."""
    print(f"Loading scene graphs from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.2
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def load_panoptic_mask(pan_seg_path: str, image_id: str, segments_info: List[Dict]) -> Dict[int, np.ndarray]:
    """
    Load panoptic segmentation mask and return a dict mapping segment_id -> binary mask.
    
    Args:
        pan_seg_path: Path to panoptic segmentation PNG
        image_id: Image ID (used for error messages)
        segments_info: List of segment info dicts with 'id' keys
    
    Returns:
        Dict mapping segment_id (int) -> binary mask (H x W numpy array)
    """
    if not os.path.exists(pan_seg_path):
        print(f"  Warning: Panoptic mask not found: {pan_seg_path}")
        return {}
    
    # Load panoptic segmentation (RGB encoded)
    pan_img = np.array(Image.open(pan_seg_path))
    
    # Decode RGB to segment IDs using PSG encoding: id = R + G * 256 + B * 256^2
    if len(pan_img.shape) == 3 and pan_img.shape[2] == 3:
        segment_map = pan_img[:, :, 0] + pan_img[:, :, 1] * 256 + pan_img[:, :, 2] * 256 * 256
    else:
        # Grayscale or single channel - use directly
        segment_map = pan_img
    
    # Extract binary masks for each segment
    masks = {}
    for seg in segments_info:
        seg_id = seg['id']
        mask = (segment_map == seg_id).astype(np.uint8)
        if mask.sum() > 0:  # Only store non-empty masks
            masks[seg_id] = mask
    
    return masks


def draw_nodes_on_image(
    image: Image.Image,
    nodes: List[Dict],
    edges: List[Dict],
    segments_info: List[Dict],
    masks: Dict[int, np.ndarray],
    title: str,
    is_prediction: bool = False
) -> Image.Image:
    """
    Draw nodes and edges on an image. Shows masks if available, otherwise just labels.
    
    Args:
        image: PIL Image
        nodes: List of node dicts with 'id' and 'label'
        edges: List of edge dicts with 'source', 'target', 'relation'
        segments_info: List of segment info dicts (for GT) or None (for predictions)
        masks: Dict mapping segment_id -> binary mask (may be empty)
        title: Title to draw at the top
        is_prediction: Whether this is prediction (True) or ground truth (False)
    
    Returns:
        PIL Image with nodes and edges overlaid
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create overlay
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Generate colors for each node
    colors = get_distinct_colors(len(nodes))
    node_colors = {node['id']: colors[idx] for idx, node in enumerate(nodes)}
    
    # Track node positions (for drawing edges later)
    node_positions = {}
    
    # Draw masks or labels
    masks_available = len(masks) > 0
    
    if masks_available:
        # Draw with masks
        for idx, node in enumerate(nodes):
            node_id = node['id']
            label = node['label']
            color = colors[idx]
            
            # Find the corresponding mask
            mask = None
            if is_prediction:
                if node_id.startswith('obj_'):
                    obj_idx = int(node_id.split('_')[1])
                    mask = masks.get(obj_idx)
            else:
                if node_id.startswith('seg_'):
                    seg_id = int(node_id.split('_')[1])
                    mask = masks.get(seg_id)
            
            if mask is not None and mask.sum() > 0:
                # Create colored overlay for this mask
                mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                mask_rgba[mask > 0] = (*color, 100)  # Semi-transparent
                
                # Paste overlay
                mask_img = Image.fromarray(mask_rgba, 'RGBA')
                overlay = Image.alpha_composite(overlay.convert('RGBA'), mask_img).convert('RGB')
                
                # Draw label at centroid
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    cy, cx = int(ys.mean()), int(xs.mean())
                    node_positions[node_id] = (cx, cy)
                    
                    # Draw text background
                    text = f"{label}"
                    bbox = draw.textbbox((cx, cy), text, font=small_font)
                    draw.rectangle([(bbox[0]-2, bbox[1]-2), (bbox[2]+2, bbox[3]+2)], fill=(0, 0, 0, 200))
                    draw.text((cx, cy), text, fill=color + (255,), font=small_font)
    else:
        # No masks available - draw list of labels on the side
        y_offset = 50
        x_offset = 10
        draw.text((x_offset, y_offset), "Detected Objects:", fill=(255, 255, 255, 255), font=font)
        y_offset += 25
        
        for idx, node in enumerate(nodes[:20]):  # Show first 20
            color = colors[idx]
            text = f"{node['id']}: {node['label']}"
            draw.text((x_offset, y_offset), text, fill=color + (255,), font=small_font)
            y_offset += 20
        
        if len(nodes) > 20:
            draw.text((x_offset, y_offset), f"... and {len(nodes) - 20} more", fill=(200, 200, 200, 255), font=small_font)
    
    # Draw title
    title_y = 10
    title_bbox = draw.textbbox((10, title_y), title, font=font)
    draw.rectangle([(title_bbox[0]-5, title_bbox[1]-2), (title_bbox[2]+5, title_bbox[3]+2)], fill=(0, 0, 0, 220))
    draw.text((10, title_y), title, fill=(255, 255, 255, 255), font=font)
    
    # Draw edge count
    edge_text = f"Edges: {len(edges)}"
    edge_bbox = draw.textbbox((image.width - 150, title_y), edge_text, font=small_font)
    draw.rectangle([(edge_bbox[0]-5, edge_bbox[1]-2), (edge_bbox[2]+5, edge_bbox[3]+2)], fill=(0, 0, 0, 220))
    draw.text((image.width - 150, title_y), edge_text, fill=(255, 255, 255, 255), font=small_font)
    
    return overlay


def visualize_image_pair(
    image_id: str,
    scene_graphs_data: Dict,
    psg_data: Dict,
    coco_images_dir: str,
    pan_seg_dir: str,
    output_dir: str
):
    """
    Visualize prediction vs ground truth for a single image.
    
    Args:
        image_id: Image ID to visualize
        scene_graphs_data: Loaded scene graphs data
        psg_data: Loaded PSG data
        coco_images_dir: Directory containing COCO images
        pan_seg_dir: Directory containing panoptic segmentation masks
        output_dir: Output directory for visualizations
    """
    print(f"\nProcessing image {image_id}...")
    
    # Get scene graph data
    if image_id not in scene_graphs_data['images']:
        print(f"  Warning: Image {image_id} not in scene graphs")
        return
    
    graphs = scene_graphs_data['images'][image_id]
    
    # Get PSG ground truth entry (image_id is string in both)
    if image_id not in psg_data['by_image']:
        print(f"  Warning: Image {image_id} not in PSG data")
        return
    
    gt_entry = psg_data['by_image'][image_id]
    
    # Find image path
    image_filename = gt_entry.get('file_name', '')
    candidates = [
        os.path.join(coco_images_dir, image_filename),
        os.path.join(coco_images_dir, os.path.basename(image_filename)),
        os.path.join(coco_images_dir, 'train2017', os.path.basename(image_filename)),
        os.path.join(coco_images_dir, 'val2017', os.path.basename(image_filename)),
    ]
    
    image_path = None
    for c in candidates:
        if os.path.exists(c):
            image_path = c
            break
    
    if image_path is None:
        print(f"  Warning: Image file not found. Tried: {candidates}")
        return
    
    print(f"  Loading image from: {image_path}")
    image = Image.open(image_path)
    
    # Load panoptic segmentation for ground truth
    pan_seg_filename = gt_entry.get('pan_seg_file_name', '')
    pan_seg_path = os.path.join(pan_seg_dir, pan_seg_filename)
    
    print(f"  Loading GT panoptic mask from: {pan_seg_path}")
    gt_masks = load_panoptic_mask(pan_seg_path, image_id, gt_entry['segments_info'])
    
    # For predictions, we need to load prediction masks if available
    # Note: The scene graph JSON doesn't contain mask data, so we'll skip prediction mask visualization
    # and only show GT masks for now
    pred_masks = {}  # Empty for now - would need prediction masks from model output
    
    # Draw GT visualization
    gt_vis = draw_nodes_on_image(
        image,
        graphs['ground_truth']['nodes'],
        graphs['ground_truth']['edges'],
        gt_entry['segments_info'],
        gt_masks,
        f"Ground Truth (Image {image_id})",
        is_prediction=False
    )
    
    # Draw prediction visualization
    pred_vis = draw_nodes_on_image(
        image,
        graphs['prediction']['nodes'],
        graphs['prediction']['edges'],
        [],
        pred_masks,
        f"Prediction (Image {image_id})",
        is_prediction=True
    )
    
    # Create side-by-side comparison
    width = image.width
    height = image.height
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(pred_vis, (0, 0))
    comparison.paste(gt_vis, (width, 0))
    
    # Save
    output_path = os.path.join(output_dir, f"vis_{image_id}.jpg")
    comparison.save(output_path, quality=90)
    print(f"  Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize scene graph predictions vs ground truth"
    )
    parser.add_argument(
        '--scene-graphs',
        type=str,
        required=True,
        help='Path to scene graphs JSON file'
    )
    parser.add_argument(
        '--psg-json',
        type=str,
        default='/home/mpws25/OneDrive_1_11-4-2025/psg/psg.json',
        help='Path to PSG annotations JSON'
    )
    parser.add_argument(
        '--coco-images',
        type=str,
        default='/home/mpws25/datasets/coco',
        help='Directory containing COCO images'
    )
    parser.add_argument(
        '--pan-seg-dir',
        type=str,
        default='/home/mpws25/OneDrive_1_11-4-2025/psg',
        help='Directory containing panoptic segmentation masks'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--images',
        type=str,
        default=None,
        help='Comma-separated list of image IDs to visualize (default: all)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to visualize'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    scene_graphs_data = load_scene_graphs(args.scene_graphs)
    psg_data = load_psg_data(args.psg_json)
    
    # Determine which images to visualize
    if args.images:
        image_ids = args.images.split(',')
    else:
        image_ids = list(scene_graphs_data['images'].keys())
    
    if args.max_images:
        image_ids = image_ids[:args.max_images]
    
    print(f"\nVisualizing {len(image_ids)} images...")
    
    # Visualize each image
    for idx, image_id in enumerate(image_ids):
        print(f"\n[{idx+1}/{len(image_ids)}]")
        try:
            visualize_image_pair(
                image_id,
                scene_graphs_data,
                psg_data,
                args.coco_images,
                args.pan_seg_dir,
                args.output_dir
            )
        except Exception as e:
            print(f"  Error visualizing image {image_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
