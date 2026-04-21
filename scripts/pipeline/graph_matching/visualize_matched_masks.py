"""
Visualize matched prediction and ground truth masks from scene graph matching results.

This script:
1. Loads scene graphs and matching results
2. Loads prediction masks (from pkl) and GT masks (from panoptic PNGs)
3. Creates side-by-side visualizations showing:
   - Matched masks with consistent colors between pred/GT
   - Unmatched pred masks in one color
   - Unmatched GT masks in another color
   - Labels and match indicators
"""

import json
import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_prediction_masks(masks_pkl_path: str) -> Dict:
    """Load prediction masks from pickle file."""
    print(f"Loading prediction masks from {masks_pkl_path}...")
    with open(masks_pkl_path, 'rb') as f:
        return pickle.load(f)


def load_gt_panoptic_mask(pan_seg_path: str, segments_info: List[Dict]) -> Dict[int, np.ndarray]:
    """
    Load ground truth panoptic segmentation and extract individual masks.
    
    Returns:
        Dict mapping segment_id -> binary mask (H x W)
    """
    if not os.path.exists(pan_seg_path):
        return {}
    
    # Load RGB-encoded panoptic mask
    pan_img = np.array(Image.open(pan_seg_path))
    
    # Decode: segment_id = R + G * 256 + B * 256^2
    if len(pan_img.shape) == 3 and pan_img.shape[2] == 3:
        segment_map = (pan_img[:, :, 0].astype(np.int32) + 
                      pan_img[:, :, 1].astype(np.int32) * 256 + 
                      pan_img[:, :, 2].astype(np.int32) * 256 * 256)
    else:
        segment_map = pan_img
    
    # Extract masks
    masks = {}
    for seg in segments_info:
        seg_id = seg['id']
        mask = (segment_map == seg_id).astype(np.uint8)
        if mask.sum() > 0:
            masks[seg_id] = mask
    
    return masks


def get_distinct_colors(n: int, saturation=0.9, value=0.95) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors using golden angle for better distribution."""
    colors = []
    golden_ratio = 0.618033988749895
    for i in range(n):
        # Use golden angle for better color distribution
        hue = (i * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_single_panel_masks(
    image: Image.Image,
    masks: Dict[str, np.ndarray],
    labels: Dict[str, str],
    colors: Dict[str, Tuple[int, int, int]],
    title: str,
    show_ids: bool = True
) -> Image.Image:
    """
    Draw masks on a single panel with labels and IDs.
    """
    img_array = np.array(image.convert('RGB'))
    overlay = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    
    label_positions = []
    
    for mask_id, mask in masks.items():
        color = colors.get(mask_id, (128, 128, 128))
        overlay[mask > 0] = (*color, 200)
        
        # Get centroid for label
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            cy, cx = int(ys.mean()), int(xs.mean())
            label = labels.get(mask_id, mask_id)
            if show_ids:
                text = f"{mask_id}: {label}"
            else:
                text = label
            label_positions.append((cx, cy, text, color))
    
    # Composite
    overlay_img = Image.fromarray(overlay, 'RGBA')
    result = Image.alpha_composite(image.convert('RGBA'), overlay_img).convert('RGB')
    
    # Draw labels
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    for cx, cy, text, color in label_positions:
        bbox = draw.textbbox((cx, cy), text, font=font)
        draw.rectangle([(bbox[0]-2, bbox[1]-2), (bbox[2]+2, bbox[3]+2)], fill=(0, 0, 0, 180))
        draw.text((cx, cy), text, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    
    # Draw title
    title_bbox = draw.textbbox((10, 10), title, font=title_font)
    draw.rectangle([(title_bbox[0]-5, title_bbox[1]-2), (title_bbox[2]+5, title_bbox[3]+2)], fill=(0, 0, 0, 230))
    draw.text((10, 10), title, fill=(255, 255, 255), font=title_font)
    
    return result


def draw_side_by_side_comparison(
    pred_image: Image.Image,
    gt_image: Image.Image,
    pred_masks: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
    pred_labels: Dict[str, str],
    gt_labels: Dict[str, str],
    matched_pairs: List[Tuple[str, str]],
    metrics: Dict[str, float]
) -> Image.Image:
    """
    Create side-by-side comparison of prediction and ground truth masks.
    Matched pairs use the same color on both sides.
    
    Args:
        pred_image: Image for prediction side (PT)
        gt_image: Image for ground truth side (GT)
    """
    # Create match mapping
    matched_pred = {}
    matched_gt = {}
    for item in matched_pairs:
        if len(item) >= 2:
            pred_id = item[0]
            gt_id = item[1]
            matched_pred[pred_id] = gt_id
            matched_gt[gt_id] = pred_id
    
    # Generate bright, distinct colors for matched pairs
    num_matched = len(matched_pairs)
    matched_colors_list = get_distinct_colors(num_matched, saturation=0.95, value=1.0)
    
    # Assign colors
    pred_colors = {}
    gt_colors = {}
    
    # Colors for matched pairs (same bright color on both sides)
    for idx, item in enumerate(matched_pairs):
        if len(item) >= 2:
            pred_id = item[0]
            gt_id = item[1]
            color = matched_colors_list[idx]
            pred_colors[pred_id] = color
            gt_colors[gt_id] = color
    
    # Black for unmatched objects (more clear distinction)
    unmatched_pred_color = (40, 40, 40)  # Dark gray/black for false positives
    unmatched_gt_color = (40, 40, 40)    # Dark gray/black for false negatives
    
    for pred_id in pred_masks:
        if pred_id not in matched_pred:
            pred_colors[pred_id] = unmatched_pred_color
    
    for gt_id in gt_masks:
        if gt_id not in matched_gt:
            gt_colors[gt_id] = unmatched_gt_color
    
    # Draw prediction panel (use PT image)
    pred_panel = draw_single_panel_masks(
        pred_image,
        pred_masks,
        pred_labels,
        pred_colors,
        f"Prediction (PT) - {len(pred_masks)} masks"
    )
    
    # Draw ground truth panel (use GT image)
    gt_panel = draw_single_panel_masks(
        gt_image,
        gt_masks,
        gt_labels,
        gt_colors,
        f"Ground Truth (GT) - {len(gt_masks)} masks"
    )
    
    # Resize panels to match prediction panel height (assuming PT is the main subject)
    if gt_panel.height != pred_panel.height:
        aspect = gt_panel.width / gt_panel.height
        target_height = pred_panel.height
        target_width = int(target_height * aspect)
        # Use getattr for compatibility with different Pillow versions
        resample_method = getattr(Image, 'Resampling', Image).LANCZOS
        gt_panel = gt_panel.resize((target_width, target_height), resample_method)

    # Combine side by side
    width = pred_panel.width + gt_panel.width
    height = pred_panel.height + 120  # Extra space for info
    combined = Image.new('RGB', (width, height), (255, 255, 255))
    
    combined.paste(pred_panel, (0, 0))
    combined.paste(gt_panel, (pred_panel.width, 0))
    
    # Draw center divider and matching lines
    draw = ImageDraw.Draw(combined)
    divider_x = pred_panel.width
    draw.line([(divider_x, 0), (divider_x, height)], fill=(0, 0, 0), width=3)
    
    # Draw info panel at bottom
    info_y = pred_panel.height
    draw.rectangle([(0, info_y), (width, height)], fill=(240, 240, 240))
    
    try:
        info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        info_font = ImageFont.load_default()
        small_font = info_font
    
    # Metrics
    metrics_text = f"Metrics: P={metrics.get('precision', 0):.3f} | R={metrics.get('recall', 0):.3f} | F1={metrics.get('f1_score', 0):.3f}"
    draw.text((20, info_y + 10), metrics_text, fill=(0, 0, 0), font=info_font)
    
    # Matching summary
    num_matched = len(matched_pairs)
    num_fp = len([p for p in pred_masks if p not in matched_pred])
    num_fn = len([g for g in gt_masks if g not in matched_gt])
    
    summary_text = f"Matched pairs: {num_matched} | False Positives: {num_fp} | False Negatives: {num_fn}"
    draw.text((20, info_y + 35), summary_text, fill=(0, 0, 0), font=small_font)
    
    # Legend with improved clarity
    legend_y = info_y + 60
    draw.text((20, legend_y), "Legend:", fill=(0, 0, 0), font=info_font)
    
    # Matched pairs - show example colors
    box_y = legend_y + 25
    if num_matched > 0:
        # Show first 3 matched colors as examples
        x_offset = 20
        for i in range(min(3, num_matched)):
            color = matched_colors_list[i]
            draw.rectangle([(x_offset, box_y), (x_offset + 20, box_y + 15)], fill=color, outline=(0, 0, 0), width=2)
            x_offset += 25
        draw.text((x_offset + 5, box_y), f"Matched pairs (same bright color on both sides) - {num_matched} total", fill=(0, 0, 0), font=small_font)
    else:
        draw.text((50, box_y), "No matched pairs", fill=(0, 0, 0), font=small_font)
    
    # Unmatched
    box_y += 25
    draw.rectangle([(20, box_y), (40, box_y + 15)], fill=unmatched_pred_color, outline=(255, 255, 255), width=2)
    draw.text((50, box_y), f"Unmatched (black/dark) - False Positives (left): {num_fp}, False Negatives (right): {num_fn}", fill=(0, 0, 0), font=small_font)
    
    return combined


def draw_masks_with_matching(
    image: Image.Image,
    pred_masks: Dict[str, np.ndarray],  # node_id -> mask
    gt_masks: Dict[str, np.ndarray],    # node_id -> mask
    pred_labels: Dict[str, str],        # node_id -> label
    gt_labels: Dict[str, str],          # node_id -> label
    matched_pairs: List[Tuple[str, str]],  # [(pred_id, gt_id), ...]
    title: str
) -> Image.Image:
    """
    Draw masks with color-coded matching.
    
    - Matched pairs: use same color for pred and GT
    - Unmatched pred: use distinct "false positive" color
    - Unmatched GT: use distinct "false negative" color
    """
    img_array = np.array(image.convert('RGB'))
    overlay = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    
    # Create match mapping (matched_pairs format is [(pred_id, gt_id, pred_label, gt_label), ...])
    matched_pred = {}
    matched_gt = {}
    for item in matched_pairs:
        if len(item) >= 2:
            # Handle both (pred, gt) and (pred, gt, pred_label, gt_label) formats
            pred_id = item[0]
            gt_id = item[1]
            matched_pred[pred_id] = gt_id
            matched_gt[gt_id] = pred_id
    
    # Generate bright, distinct colors for matched pairs
    num_matched = len(matched_pairs)
    matched_colors = get_distinct_colors(num_matched, saturation=0.95, value=1.0)
    
    # Black for unmatched (clear distinction from matched)
    unmatched_pred_color = (40, 40, 40)  # Dark gray/black for false positives
    unmatched_gt_color = (40, 40, 40)    # Dark gray/black for false negatives
    
    # Track centroids for labels
    label_positions = []
    
    # Draw matched masks (both pred and GT with same color)
    for idx, item in enumerate(matched_pairs):
        # Handle both tuple and list formats
        if len(item) >= 2:
            pred_id = item[0]
            gt_id = item[1]
        else:
            continue
            
        color = matched_colors[idx]
        
        # Draw prediction mask
        if pred_id in pred_masks:
            mask = pred_masks[pred_id]
            overlay[mask > 0] = (*color, 120)
            
            # Compute centroid
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                pred_label = pred_labels.get(pred_id, pred_id)
                gt_label = gt_labels.get(gt_id, gt_id)
                label_positions.append((cx, cy, f"✓ {pred_label} → {gt_label}", color))
        
        # Draw GT mask outline (to show alignment)
        if gt_id in gt_masks:
            mask = gt_masks[gt_id]
            # Draw border
            from scipy import ndimage
            dilated = ndimage.binary_dilation(mask, iterations=2)
            border = dilated & ~mask.astype(bool)
            overlay[border] = (*color, 200)
    
    # Draw unmatched prediction masks
    for pred_id, mask in pred_masks.items():
        if pred_id not in matched_pred:
            overlay[mask > 0] = (*unmatched_pred_color, 100)
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                label = pred_labels.get(pred_id, pred_id)
                label_positions.append((cx, cy, f"✗ {label} (FP)", unmatched_pred_color))
    
    # Draw unmatched GT masks
    for gt_id, mask in gt_masks.items():
        if gt_id not in matched_gt:
            overlay[mask > 0] = (*unmatched_gt_color, 100)
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                label = gt_labels.get(gt_id, gt_id)
                label_positions.append((cx, cy, f"✗ {label} (FN)", unmatched_gt_color))
    
    # Composite overlay onto image
    overlay_img = Image.fromarray(overlay, 'RGBA')
    result = Image.alpha_composite(image.convert('RGBA'), overlay_img).convert('RGB')
    
    # Draw labels
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    for cx, cy, text, color in label_positions:
        # Background box with semi-transparent black
        bbox = draw.textbbox((cx, cy), text, font=font)
        draw.rectangle([(bbox[0]-3, bbox[1]-3), (bbox[2]+3, bbox[3]+3)], fill=(0, 0, 0, 220))
        # Use white text for better contrast, with colored outline matching the mask
        draw.text((cx, cy), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=color)
    
    # Draw title
    title_bbox = draw.textbbox((10, 10), title, font=title_font)
    draw.rectangle([(title_bbox[0]-5, title_bbox[1]-2), (title_bbox[2]+5, title_bbox[3]+2)], fill=(0, 0, 0, 220))
    draw.text((10, 10), title, fill=(255, 255, 255), font=title_font)
    
    # Draw improved legend
    legend_y = result.height - 90
    legend_width = 350
    draw.rectangle([(5, legend_y-5), (legend_width, result.height-5)], fill=(255, 255, 255, 230))
    draw.rectangle([(5, legend_y-5), (legend_width, result.height-5)], outline=(0, 0, 0), width=2)
    
    draw.text((10, legend_y), "Legend:", fill=(0, 0, 0), font=title_font)
    
    # Show example matched colors
    if len(matched_colors) > 0:
        y_pos = legend_y + 25
        draw.text((10, y_pos), "Matched pairs (bright colors):", fill=(0, 0, 0), font=font)
        x_pos = 15
        for i in range(min(5, len(matched_colors))):
            color = matched_colors[i]
            draw.rectangle([(x_pos, y_pos + 18), (x_pos + 15, y_pos + 33)], fill=color, outline=(0, 0, 0), width=2)
            x_pos += 20
    
    # Unmatched
    y_pos = legend_y + 55
    draw.rectangle([(15, y_pos), (30, y_pos + 15)], fill=unmatched_pred_color, outline=(200, 200, 200), width=2)
    draw.text((40, y_pos), "Unmatched (black/dark)", fill=(0, 0, 0), font=font)
    
    return result


def visualize_matched_image(
    image_id: str,
    scene_graphs: Dict,
    matching_results: Dict,
    pred_masks_data: Dict,
    psg_data: Dict,
    coco_images_dir: str,
    panoptic_dir: str,
    output_dir: str,
    vis_mode: str = 'both',
    pt_images_dir: str = None,
    pt_panoptic_dir: str = None,
    gt_anno_by_image: Dict = None
) -> None:
    """Visualize one image with matched masks."""
    print(f"\nProcessing image {image_id}...")
    
    # Get scene graph data
    if image_id not in scene_graphs['images']:
        print(f"  Skipping: not in scene graphs")
        return
    
    graphs = scene_graphs['images'][image_id]
    
    # Get matching results
    if image_id not in matching_results['per_image_results']:
        print(f"  Skipping: no matching results")
        return
    
    match_result = matching_results['per_image_results'][image_id]
    if match_result is None:
        print(f"  Skipping: matching failed")
        return
    
    # Get PSG entry for image path (try both string and int keys)
    gt_entry = None
    if str(image_id) in psg_data['by_image']:
        gt_entry = psg_data['by_image'][str(image_id)]
    elif int(image_id) in psg_data['by_image']:
        gt_entry = psg_data['by_image'][int(image_id)]
    
    if gt_entry is None:
        # Try fallback locations in the local workflow directories
        fallback_paths = [
            os.path.join(os.getcwd(), '3_descriptions_gt', f'scene-graph-description_{image_id}.json'),
            os.path.join(os.getcwd(), '2_scene_graphs_gt', f'scene-graph_{image_id}.json'),
            os.path.join(os.getcwd(), '3_descriptions', f'scene-graph-description_{image_id}.json'),
            os.path.join(os.getcwd(), '2_scene_graphs', f'scene-graph_{image_id}.json'),
            os.path.join(os.path.dirname(__file__), '..', 'full_workflow', '3_descriptions_gt', f'scene-graph-description_{image_id}.json'),
            os.path.join(os.path.dirname(__file__), '..', 'full_workflow', '2_scene_graphs_gt', f'scene-graph_{image_id}.json'),
            os.path.join(os.path.dirname(__file__), '..', 'full_workflow', '3_descriptions', f'scene-graph-description_{image_id}.json')
        ]
        for fp in fallback_paths:
            try:
                fp = os.path.abspath(fp)
                if os.path.exists(fp):
                    print(f"  Found fallback PSG entry at: {fp}")
                    with open(fp, 'r') as f:
                        gt_entry = json.load(f)
                    break
            except Exception:
                continue

    if gt_entry is None:
        print(f"  Skipping: not in PSG data and no fallback found")
        return
    
    # Load GT image
    gt_image_filename = gt_entry['file_name']
    
    # Try to get PT image filename from prediction scene graph
    # Check if we have separate PT descriptions/scene graphs
    pt_image_filename = None
    pt_entry = None
    
    # Try to extract PT image filename from matching results metadata if available
    if 'metadata' in matching_results and 'pt_dir' in matching_results['metadata']:
        pt_dir_meta = matching_results['metadata']['pt_dir'] 
        # But this is just directory. We need filename.
    
    # Try to find PT scene graph file
    pt_fallback_paths = [
        os.path.join(os.getcwd(), '3_descriptions_pt', f'scene-graph-description_{image_id}.json'),
        os.path.join(os.getcwd(), '2_scene_graphs_pt', f'scene-graph_{image_id}.json'),
        os.path.join(os.path.dirname(__file__), '..', 'full_workflow', '3_descriptions_pt', f'scene-graph-description_{image_id}.json'),
        os.path.join(os.path.dirname(__file__), '..', 'full_workflow', '2_scene_graphs_pt', f'scene-graph_{image_id}.json'),
    ]
    
    # Also look in the directory provided in scene_graphs_for_matching.json metadata
    if 'metadata' in scene_graphs and 'pt_dir' in scene_graphs['metadata']:
         pt_dir_sg = scene_graphs['metadata']['pt_dir']
         pt_fallback_paths.append(os.path.join(pt_dir_sg, f"scene-graph_{image_id}.json"))
         pt_fallback_paths.append(os.path.join(pt_dir_sg, f"scene-graph-description_{image_id}.json"))

    for fp in pt_fallback_paths:
        try:
            fp = os.path.abspath(fp)
            if os.path.exists(fp):
                # print(f"  Found PT scene graph at: {fp}") # Too verbose?
                with open(fp, 'r') as f:
                    pt_entry = json.load(f)
                    pt_image_filename = pt_entry.get('file_name', '')
                    if pt_image_filename:
                        print(f"  Found PT image filename in graph: {pt_image_filename}")
                        break
        except Exception:
            continue
    
    # If no PT entry found, try to extract from prediction scene graph metadata
    if not pt_image_filename:
        # Check scene_graphs metadata for pt_dir
        if 'metadata' in scene_graphs and 'pt_dir' in scene_graphs['metadata']:
            pt_dir = scene_graphs['metadata']['pt_dir']
            # Construct PT filename based on GT filename
            # Heuristic: replace 'gt' in path with 'pt' or looks for generated_images
            if 'gt/' in gt_image_filename:
                pt_image_filename = gt_image_filename.replace('gt/', 'pt/')
            elif '/gt/' in gt_image_filename:
                pt_image_filename = gt_image_filename.replace('/gt/', '/pt/')
            else:
                 # If GT is just filename, maybe PT is same filename but in pt_dir?
                 # But we need full path relative to what?
                 # Usually filenames are relative to some root.
                 pass
    
    # If still not found, check if we can guess it from the run configuration logic
    # The user might have images in `generated_images/<model_name>/...`
    # But we don't know the model name easily here without parsing more config.
    
    # If still not found, check if we can guess it from the run configuration logic
    # The user might have images in `generated_images/<model_name>/...`
    # But we don't know the model name easily here without parsing more config.
    
    # Try generic search in likely PT image directories if we are in a workflow run
    if not pt_image_filename:
         # Look for "generated_images" in parent directories
         workflow_dir = os.path.dirname(os.path.dirname(output_dir)) if output_dir else os.getcwd()
         # This is hard because we don't know the specific model subfolder.
         pass
    
    # Helper function to load image
    def load_image_from_filename(filename, preferred_dir=None):
        if not filename:
             return None
             
        # Clean filename
        if filename.startswith('datasets/'):
            filename_clean = filename[len('datasets/'):]
        else:
            filename_clean = filename
        
        candidates = []
        
        if preferred_dir:
            candidates.extend([
                os.path.join(preferred_dir, filename_clean),
                os.path.join(preferred_dir, filename),
                os.path.join(preferred_dir, os.path.basename(filename)),
            ])
            
        candidates.extend([
            os.path.join(coco_images_dir, filename_clean),
            os.path.join(coco_images_dir, filename),
            os.path.join(coco_images_dir, os.path.basename(filename)),
            os.path.join(coco_images_dir, 'val2017', os.path.basename(filename)),
            os.path.join(coco_images_dir, 'train2017', os.path.basename(filename)),
        ])
        
        for c in candidates:
            if os.path.exists(c):
                return c
        
        # Fallback
        extra_img_dir = '/sc/home/anton.hackl/master-project/vlm-benchmark/psg_generation/images'
        fallback_img = os.path.join(extra_img_dir, os.path.basename(filename))
        if os.path.exists(fallback_img):
            return fallback_img
        
        # Search in generated_images subdirectories
        datasets_dir = os.path.dirname(coco_images_dir.rstrip('/'))
        generated_images_dir = os.path.join(datasets_dir, 'generated_images')
        
        # Also try relative to workflow
        if not os.path.exists(generated_images_dir):
             generated_images_dir = os.path.join(os.getcwd(), 'datasets', 'generated_images')

        if os.path.exists(generated_images_dir):
            for subdir in sorted(os.listdir(generated_images_dir)):
                candidate = os.path.join(generated_images_dir, subdir, filename_clean)
                if os.path.exists(candidate):
                    return candidate
                candidate = os.path.join(generated_images_dir, subdir, os.path.basename(filename))
                if os.path.exists(candidate):
                    return candidate

        return None
    
    # Load GT image (from COCO dir)
    gt_image_path = load_image_from_filename(gt_image_filename, preferred_dir=None)
    if not gt_image_path:
        print(f"  Skipping: GT image not found (tried: {gt_image_filename})")
        return
    
    print(f"  Loading GT image: {gt_image_path}")
    gt_image = Image.open(gt_image_path)
    
    # Load PT image (use GT image as fallback if PT not found)
    pt_image = None
    
    # If we have a directory but no filename, try to find by ID
    if pt_images_dir and not pt_image_filename:
        # Try common extensions
        for ext in ['.jpg', '.png', '.jpeg']:
            # Try full ID, stripped ID
            trials = [
                f"{image_id}{ext}",
                f"{int(image_id)}{ext}" if str(image_id).isdigit() else f"{image_id}{ext}",
                os.path.basename(gt_image_filename)
            ]
            for trial in trials:
                path = os.path.join(pt_images_dir, trial)
                if os.path.exists(path):
                    pt_image_filename = trial
                    print(f"  Found PT image by ID in directory: {trial}")
                    break
            if pt_image_filename: break

    if pt_image_filename:
        pt_image_path = load_image_from_filename(pt_image_filename, preferred_dir=pt_images_dir)
        if pt_image_path:
            print(f"  Loading PT image: {pt_image_path}")
            pt_image = Image.open(pt_image_path)
        else:
            print(f"  Warning: PT image not found ({pt_image_filename})")
    else:
        print(f"  Warning: No PT image filename found")
    
    if pt_image is None:
         # Try to find it in generated_images even if we fail to map it neatly
         if not pt_image_filename:
             pt_image_filename = gt_image_filename
         
         pt_image_path = load_image_from_filename(pt_image_filename, preferred_dir=None)
         if pt_image_path and pt_image_path != gt_image_path:
              print(f"  Found potential PT image at: {pt_image_path}")
              pt_image = Image.open(pt_image_path)
    
    if pt_image is None:
        print("  Using GT image for both sides (PT image missing)")
        pt_image = gt_image

    image = pt_image
    
    # Load prediction masks
    pred_masks = {}
    
    # Try to load from PT panoptic directory first (if available)
    if pt_panoptic_dir:
        # Check if this is a merged segmentation (from graph merge step)
        is_merged_seg_pt = 'merged_segmentations' in pt_panoptic_dir
        
        # Try multiple PT panoptic file locations
        pt_pan_seg_candidates = [
            os.path.join(pt_panoptic_dir, f"seg_{image_id}.png"),
            os.path.join(pt_panoptic_dir, f"seg_{int(image_id):012d}.png"),
            os.path.join(pt_panoptic_dir, f"{image_id}.png"),
        ]
        
        pt_pan_seg_path = None
        for candidate in pt_pan_seg_candidates:
            if os.path.exists(candidate) and os.path.isfile(candidate):
                pt_pan_seg_path = candidate
                break
        
        if pt_pan_seg_path:
            print(f"  Loading PT masks from panoptic: {pt_pan_seg_path}")
            
            if is_merged_seg_pt:
                print(f"  Loading PT merged segmentation (node indices map directly to pixel values)")
                # For merged segmentations, pixel values directly correspond to node indices
                pan_img = np.array(Image.open(pt_pan_seg_path))
                
                # Decode RGB to segment IDs
                if len(pan_img.shape) == 3 and pan_img.shape[2] == 3:
                    segment_map = (pan_img[:, :, 0].astype(np.int32) + 
                                  pan_img[:, :, 1].astype(np.int32) * 256 + 
                                  pan_img[:, :, 2].astype(np.int32) * 256 * 256)
                else:
                    segment_map = pan_img.astype(np.int32)
                
                # Get unique segment IDs (these should correspond to node indices 0, 1, 2...)
                unique_seg_ids = sorted([sid for sid in np.unique(segment_map) if sid != 0])
                
                # Create masks directly in pred_masks
                for node_idx, node in enumerate(graphs['prediction']['nodes']):
                    node_id = node['id']  # e.g., "node_0", "node_1"
                    # For merged masks, node_0 -> first non-zero segment ID, node_1 -> second, etc.
                    if node_idx < len(unique_seg_ids):
                        seg_id = unique_seg_ids[node_idx]
                        mask = (segment_map == seg_id).astype(np.uint8)
                        if mask.sum() > 0:
                            pred_masks[node_id] = mask
            else:
                # Non-merged PT segmentation - would need segments_info
                # This is less common but could be supported similarly to GT
                print(f"  Warning: Non-merged PT panoptic segmentations not fully supported yet")
    
    # Fallback: Load from pickle file if no PT panoptic masks or if PT masks dict is still empty
    if not pred_masks and image_id in pred_masks_data:
        pred_mask_info = pred_masks_data[image_id]
        pred_masks_array = pred_mask_info['masks']  # (N, H, W)
        pred_node_ids = pred_mask_info['node_ids']
        
        # Resize prediction masks to image size if needed
        if pred_masks_array.shape[1:] != (image.height, image.width):
            from scipy.ndimage import zoom
            scale_h = image.height / pred_masks_array.shape[1]
            scale_w = image.width / pred_masks_array.shape[2]
            pred_masks_resized = []
            for mask in pred_masks_array:
                resized = zoom(mask, (scale_h, scale_w), order=0) > 0.5
                pred_masks_resized.append(resized.astype(np.uint8))
            pred_masks_array = np.array(pred_masks_resized)
        
        pred_masks = {pred_node_ids[i]: pred_masks_array[i] for i in range(len(pred_node_ids))}
    
    # Check if we have any PT masks
    if not pred_masks:
        print(f"  Skipping: no prediction masks (checked both PT panoptic and pickle)")
        return
    
    
    # Try to load GT masks from prediction masks pickle first (if GT masks were stored)
    gt_masks = {}
    
    # Check if GT masks are in the prediction masks data (for GT vs PT comparisons)
    # The masks pickle might have both GT and PT masks for the same image_id
    if image_id in pred_masks_data:
        # Check if there's a separate GT entry or if we need to load from scene graph directory
        pass  # Will try panoptic first, then fallback to scene graph masks
    
    # Load GT panoptic masks
    pan_seg_filename = gt_entry.get('pan_seg_file_name', '')
    if not pan_seg_filename and 'file_name' in gt_entry:
        pan_seg_filename = os.path.splitext(gt_entry['file_name'])[0] + '.png'

    # Try multiple locations
    pan_seg_candidates = [
        os.path.join(panoptic_dir, pan_seg_filename),
        os.path.join(panoptic_dir, os.path.basename(pan_seg_filename)),
        os.path.join(panoptic_dir, 'panoptic_val2017', os.path.basename(pan_seg_filename)),
        os.path.join(panoptic_dir, 'panoptic_train2017', os.path.basename(pan_seg_filename)),
        # Add handling for 'seg_' prefix and zero-padding variations
        os.path.join(panoptic_dir, f"seg_{os.path.basename(pan_seg_filename)}"),
        os.path.join(panoptic_dir, f"seg_{image_id}.png"), 
        os.path.join(panoptic_dir, f"seg_{int(image_id):012d}.png"),
    ]
    
    pan_seg_path = None
    for candidate in pan_seg_candidates:
        if os.path.exists(candidate) and os.path.isfile(candidate):
            pan_seg_path = candidate
            break
    
    gt_masks_by_seg_id = {}
    if pan_seg_path:
        print(f"  Loading GT masks: {pan_seg_path}")
        
        # First try to get segments_info from gt_anno_by_image
        segments_info = None
        if gt_anno_by_image and int(image_id) in gt_anno_by_image:
            anno_entry = gt_anno_by_image[int(image_id)]
            if 'segments_info' in anno_entry:
                segments_info = anno_entry['segments_info']
                print(f"  Using segments_info from anno.json ({len(segments_info)} segments)")
        
        # Fallback: check if segments_info exists in GT entry
        if not segments_info and gt_entry and 'segments_info' in gt_entry:
            segments_info = gt_entry['segments_info']
            print(f"  Using segments_info from GT entry ({len(segments_info)} segments)")
        
        # Check if this is a merged segmentation (from graph merge step)
        is_merged_seg = 'merged_segmentations' in pan_seg_path
        
        if is_merged_seg:
            print(f"  Loading merged segmentation (node indices map directly to pixel values)")
            # For merged segmentations, pixel values directly correspond to node indices
            pan_img = np.array(Image.open(pan_seg_path))
            
            # Decode RGB to segment IDs
            if len(pan_img.shape) == 3 and pan_img.shape[2] == 3:
                segment_map = (pan_img[:, :, 0].astype(np.int32) + 
                              pan_img[:, :, 1].astype(np.int32) * 256 + 
                              pan_img[:, :, 2].astype(np.int32) * 256 * 256)
            else:
                segment_map = pan_img.astype(np.int32)
            
            # Get unique segment IDs (these should correspond to node indices 0, 1, 2...)
            unique_seg_ids = sorted([sid for sid in np.unique(segment_map) if sid != 0])
            
            # Create masks directly in gt_masks (not gt_masks_by_seg_id)
            for node_idx, node in enumerate(graphs['ground_truth']['nodes']):
                node_id = node['id']  # e.g., "node_0", "node_1"
                # For merged masks, node_0 -> first non-zero segment ID, node_1 -> second, etc.
                if node_idx < len(unique_seg_ids):
                    seg_id = unique_seg_ids[node_idx]
                    mask = (segment_map == seg_id).astype(np.uint8)
                    if mask.sum() > 0:
                        gt_masks[node_id] = mask  # Directly populate gt_masks
        elif segments_info:
            gt_masks_by_seg_id = load_gt_panoptic_mask(pan_seg_path, segments_info)
        else:
            print(f"  Warning: No segments_info available, attempting to deduce masks from indices")
            try:
                # Fallback: Load panoptic image and assume pixel values correspond to object indices
                pan_img = np.array(Image.open(pan_seg_path))
                
                # Debug unique values
                unique_vals = np.unique(pan_img)
                if len(unique_vals) < 20: 
                    print(f"    Panoptic image unique values: {unique_vals}")
                
                if len(pan_img.shape) == 3 and pan_img.shape[2] == 3:
                     # RGB to ID
                    raw_id_map = (pan_img[:, :, 0].astype(np.int32) + 
                                  pan_img[:, :, 1].astype(np.int32) * 256 + 
                                  pan_img[:, :, 2].astype(np.int32) * 256 * 256)
                else:
                    raw_id_map = pan_img.astype(np.int32)
                
                # Check scene graph nodes for indices
                for node in graphs['ground_truth']['nodes']:
                    # Try to find an index for this node
                    obj_idx = -1
                    if 'index' in node:
                        obj_idx = int(node['index'])
                        # If values are large (RGB encoded), maybe index isn't directly the value?
                        # But typically 'index' in scene graph comes from segmentation ID
                        pass
                    elif node['id'].startswith('node_'):
                        try:
                            obj_idx = int(node['id'].split('_')[1])
                        except: pass
                    
                    if obj_idx >= 0:
                        # Try to find this index in the map
                        mask = (raw_id_map == obj_idx).astype(np.uint8)
                        if mask.sum() == 0:
                            # Try index + 1? Some schemes use 0 for BG
                            mask = (raw_id_map == (obj_idx + 1)).astype(np.uint8)
                        
                        if mask.sum() > 0:
                            gt_masks_by_seg_id[obj_idx] = mask
                            # Also map explicitly if we can
                            gt_masks[node['id']] = mask
            except Exception as e:
                print(f"  Failed to deduce masks from panoptic image: {e}")

    # If no panoptic masks, try to load from GT scene graph directory
    # Check both gt_masks_by_seg_id and gt_masks (merged segmentations populate gt_masks directly)
    if not gt_masks_by_seg_id and not gt_masks:
        # Try to load GT masks from scene graph pickle files
        # Build possible paths for GT scene graph directory
        base_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.dirname(output_dir)) if output_dir else os.getcwd(),  # Go up from 6_visualizations to full_workflow
            os.path.join(os.path.dirname(__file__), '..', 'full_workflow'),
        ]
        
        gt_scene_graph_dirs = []
        for base in base_dirs:
            base = os.path.abspath(base)
            gt_scene_graph_dirs.extend([
                os.path.join(base, '2_scene_graphs_gt'),
                os.path.join(os.path.dirname(base), 'full_workflow', '2_scene_graphs_gt'),
            ])
        
        # Remove duplicates
        seen = set()
        for sg_dir in gt_scene_graph_dirs:
            sg_dir = os.path.abspath(sg_dir)
            if sg_dir in seen:
                continue
            seen.add(sg_dir)
            
            pkl_path = os.path.join(sg_dir, 'scene-graph.pkl')
            if os.path.exists(pkl_path):
                try:
                    print(f"  Loading GT masks from scene graph: {pkl_path}")
                    import pickle
                    with open(pkl_path, 'rb') as f:
                        pkl_data = pickle.load(f)
                    
                    # Find entry for this image_id
                    # Handle both integer and string IDs in pkl data
                    mask_entry = None
                    target_id_str = str(image_id)
                    target_id_int = int(image_id) if str(image_id).isdigit() else -1
                    
                    if isinstance(pkl_data, dict):
                         # If it's a dict mapping ID -> data
                        if target_id_str in pkl_data:
                            mask_entry = pkl_data[target_id_str]
                        elif target_id_int in pkl_data:
                            mask_entry = pkl_data[target_id_int]
                    elif isinstance(pkl_data, list):
                        # If it's a list of entries
                        for entry in pkl_data:
                            entry_id = entry.get('img_id') or entry.get('image_id')
                            if str(entry_id) == target_id_str or entry_id == target_id_int:
                                mask_entry = entry
                                break
                    
                    if mask_entry:
                        mask = mask_entry.get('mask')
                        if mask is not None:
                            H, W = mask.shape
                            
                            # Resize if needed
                            if (H, W) != (image.height, image.width):
                                from scipy.ndimage import zoom
                                scale_h = image.height / H
                                scale_w = image.width / W
                                mask = zoom(mask, (scale_h, scale_w), order=0)
                            
                            print(f"    Found mask with unique values: {np.unique(mask)}")
                            
                            # Extract individual masks for each GT node
                            for node in graphs['ground_truth']['nodes']:
                                node_id = node['id']  # e.g., "node_1"
                                if node_id.startswith('node_'):
                                    try:
                                        obj_idx = int(node_id.split('_')[1])
                                        # Important: Check if obj_idx corresponds to what's in the mask
                                        # Some pipelines might use 1-based indexing in masks?
                                        # Or node indices might not match mask values directly? 
                                        # Assuming direct correspondence for now but checking
                                        
                                        gt_mask = (mask == obj_idx).astype(np.uint8)
                                        if gt_mask.sum() > 0:  # Only add if mask has pixels
                                            # Resize mask to match GT image dimensions
                                            if gt_mask.shape != (gt_image.height, gt_image.width):
                                                from scipy.ndimage import zoom
                                                scale_h = gt_image.height / gt_mask.shape[0]
                                                scale_w = gt_image.width / gt_mask.shape[1]
                                                gt_mask = zoom(gt_mask, (scale_h, scale_w), order=0).astype(np.uint8)
                                            
                                            gt_masks[node_id] = gt_mask
                                        
                                        # Also try checking if mask has index+1 (if 0 is background)
                                        # but usually node_0 -> 0 or node_0 -> 1?
                                        # Let's trust exact match for now, or maybe check 'index' attribute if avail
                                        
                                    except ValueError:
                                        pass
                            
                            print(f"    Loaded {len(gt_masks)} GT masks from scene graph")
                            break
                        else:
                             print(f"    Entry found but no 'mask' field")
                    else:
                        print(f"    Image ID {image_id} not found in {pkl_path} (checked {len(pkl_data)} entries)")

                except Exception as e:
                    print(f"    Warning: Failed to load GT masks from {pkl_path}: {e}")
                    # import traceback
                    # traceback.print_exc()
                    continue
            
            # Break outer loop if we found masks
            if gt_masks:
                break
    
    # Map GT node IDs from panoptic masks (if available)
    if gt_masks_by_seg_id and gt_anno_by_image and int(image_id) in gt_anno_by_image:
        # Get the segments_info in their original order (not sorted by ID)
        # This order should match the node indices in the scene graph
        anno_entry = gt_anno_by_image[int(image_id)]
        segments_info_ordered = anno_entry.get('segments_info', [])
        
        for node in graphs['ground_truth']['nodes']:
            node_id = node['id']  # e.g., "node_0", "node_1"
            
            # Try to find segment ID by node ID format
            if node_id.startswith('seg_'):
                # Already has segment ID format
                seg_id = int(node_id.split('_')[1])
                if seg_id in gt_masks_by_seg_id:
                    gt_masks[node_id] = gt_masks_by_seg_id[seg_id]
            elif node_id.startswith('node_'):
                # Map node index to segment ID using segments_info original order
                try:
                    node_idx = int(node_id.split('_')[1])
                    # Map by position using original segments_info order
                    if node_idx < len(segments_info_ordered):
                        seg_id = segments_info_ordered[node_idx]['id']
                        if seg_id in gt_masks_by_seg_id:
                            gt_masks[node_id] = gt_masks_by_seg_id[seg_id]
                except (ValueError, IndexError, KeyError):
                    pass
    
    # Get labels
    pred_labels = {node['id']: node['label'] for node in graphs['prediction']['nodes']}
    gt_labels = {node['id']: node['label'] for node in graphs['ground_truth']['nodes']}
    
    # Get matched node pairs from matching results
    matched_pairs = match_result.get('matched_node_pairs', [])
    
    print(f"  Drawing visualization...")
    print(f"    Pred masks: {len(pred_masks)}")
    print(f"    GT masks: {len(gt_masks)}")
    print(f"    Matched pairs: {len(matched_pairs)}")
    
    # Create metrics dict
    
    # (original and GT paths will be printed after generating visualizations)
    metrics = {
        'precision': match_result['precision'],
        'recall': match_result['recall'],
        'f1_score': match_result['f1_score']
    }
    # Always generate side-by-side and overlay visualizations, save them, and print concise paths
    side_by_side_img = draw_side_by_side_comparison(
        pt_image,  # Prediction side uses PT image
        gt_image,  # GT side uses GT image
        pred_masks,
        gt_masks,
        pred_labels,
        gt_labels,
        matched_pairs,
        metrics
    )
    side_by_side_path = os.path.join(output_dir, f"comparison_{image_id}.jpg")
    side_by_side_img.save(side_by_side_path, quality=95)

    # For overlay, use PT image (prediction side)
    # overlay_img = draw_masks_with_matching(
    #     pt_image,
    #     pred_masks,
    #     gt_masks,
    #     pred_labels,
    #     gt_labels,
    #     matched_pairs,
    #     f"Image {image_id} - P:{match_result['precision']:.2f} R:{match_result['recall']:.2f} F1:{match_result['f1_score']:.2f}"
    # )
    # overlay_path = os.path.join(output_dir, f"overlay_{image_id}.jpg")
    # overlay_img.save(overlay_path, quality=90)

    # Print only the required paths as concise strings (original and GT panoptic)
    print(f"PATH_PT_IMAGE={pt_image_path if 'pt_image_path' in dir() else 'NOT_FOUND'}")
    print(f"PATH_GT_IMAGE={gt_image_path}")
    print(f"PATH_GT_PANOPTIC={pan_seg_path if 'pan_seg_path' in dir() and pan_seg_path else 'NOT_FOUND'}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize matched prediction and ground truth masks"
    )
    parser.add_argument('--scene-graphs', type=str, required=True, help='Scene graphs JSON')
    parser.add_argument('--matching-results', type=str, required=True, help='Matching results JSON')
    parser.add_argument('--pred-masks', type=str, required=False, default=None, help='Prediction masks pickle file (optional)')
    parser.add_argument('--psg-json', type=str, default='/home/mpws25/OneDrive_1_11-4-2025/psg/psg.json')
    parser.add_argument('--coco-images', type=str, default='/home/mpws25/datasets/coco/coco')
    parser.add_argument('--panoptic-dir', type=str, default='/home/mpws25/datasets/coco/coco')
    parser.add_argument('--gt-anno-json', type=str, default=None, help='GT anno.json with segments_info (optional, auto-detected from panoptic-dir)')
    parser.add_argument('--output-dir', type=str, default='./visualizations_with_masks')
    parser.add_argument('--images', type=str, default=None, help='Comma-separated image IDs')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--vis-mode', type=str, default='both', 
                       choices=['side-by-side', 'overlay', 'both'],
                       help='Visualization mode: side-by-side, overlay, or both')
    parser.add_argument('--pt-images', type=str, default=None, help='Directory containing prediction images')
    parser.add_argument('--pt-panoptic-dir', type=str, default=None, help='Directory containing PT panoptic segmentation PNGs (e.g., merged_segmentations from graph_merge_pt)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    scene_graphs = load_json(args.scene_graphs)
    matching_results = load_json(args.matching_results)
    if args.pred_masks:
        try:
            pred_masks_data = load_prediction_masks(args.pred_masks)
        except Exception as e:
            print(f"Warning: Failed to load prediction masks '{args.pred_masks}': {e}")
            pred_masks_data = {}
    else:
        print("Warning: No prediction masks provided (--pred-masks). Proceeding with empty prediction masks.")
        pred_masks_data = {}

    psg_data = load_json(args.psg_json)
    
    # Create lookup for PSG
    psg_by_image = {}
    for entry in psg_data['data']:
        psg_by_image[entry['image_id']] = entry
    psg_data['by_image'] = psg_by_image
    
    # Load GT anno.json if available (contains segments_info for panoptic masks)
    gt_anno_data = None
    gt_anno_by_image = {}
    if args.gt_anno_json:
        gt_anno_path = args.gt_anno_json
    elif args.panoptic_dir:
        # Try to auto-detect anno.json in panoptic_dir
        gt_anno_path = os.path.join(args.panoptic_dir, 'anno.json')
    else:
        gt_anno_path = None
        
    if gt_anno_path and os.path.exists(gt_anno_path):
        print(f"Loading GT annotations from: {gt_anno_path}")
        gt_anno_data = load_json(gt_anno_path)
        # Create lookup by image_id
        for entry in gt_anno_data['data']:
            gt_anno_by_image[entry['image_id']] = entry
        print(f"  Loaded annotations for {len(gt_anno_by_image)} images")
    else:
        print(f"Warning: No GT anno.json found. GT masks may not load correctly.")
        if gt_anno_path:
            print(f"  Tried: {gt_anno_path}")
    
    # Get image IDs to process
    if args.images:
        image_ids = args.images.split(',')
    else:
        # Default to intersection of available data
        image_ids = list(matching_results['per_image_results'].keys())
        # Sort for deterministic order
        image_ids.sort()
        
        if args.max_images:
            image_ids = image_ids[:args.max_images]
    
    total_images = len(image_ids)
    print(f"Processing {total_images} images...")
    
    for i, image_id in enumerate(image_ids):
        print(f"\n[{i+1}/{total_images}]")
        visualize_matched_image(
            image_id,
            scene_graphs,
            matching_results,
            pred_masks_data,
            psg_data,
            args.coco_images,
            args.panoptic_dir,
            args.output_dir,
            args.vis_mode,
            pt_images_dir=args.pt_images,
            pt_panoptic_dir=args.pt_panoptic_dir,
            gt_anno_by_image=gt_anno_by_image
        )
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
