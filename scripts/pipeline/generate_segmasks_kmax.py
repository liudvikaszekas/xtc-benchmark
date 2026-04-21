#!/usr/bin/env python3
"""
Generate Segmentation Masks using kMaX-DeepLab

This script generates panoptic segmentation masks using kMaX-DeepLab.
Run this in a separate conda environment with the required dependencies.
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add segmentation module to path
sys.path.insert(0, str(Path(__file__).parent))
from segmentation.kmax_wrapper import KMaxSegmentor


MAX_SEG_ID = 256**3 - 1


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def mask_to_boxes(mask):
    y, x = torch.where(mask != 0)
    return torch.stack([torch.min(x), torch.min(y), torch.max(x), torch.max(y)])


def generate_segmasks_kmax(img_dir, out_dir, psg_path, kmax_config, max_images=None):
    """Generate segmentation masks using kMaX-DeepLab model."""
    print("\n=== Generating Segmentation Masks (kMaX-DeepLab) ===")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with open(psg_path) as f:
        psg_metadata = {
            k: v
            for k, v in json.load(f).items()
            if k in ("thing_classes", "stuff_classes", "predicate_classes")
        }
    
    # Initialize kMaX segmentor
    segmentor = KMaxSegmentor(kmax_config)
    segmentor.load_model()
    
    # Look for both .jpg and .png images recursively
    img_dir_path = Path(img_dir)
    img_paths = sorted(list(img_dir_path.glob("**/*.jpg")) + list(img_dir_path.glob("**/*.png")))
    
    # Limit to max_images if specified
    if max_images is not None:
        img_paths = img_paths[:max_images]
        print(f"Limited to first {max_images} images")

    # JSONL Setup
    jsonl_path = out_dir / "segmentation.jsonl"
    try:
        from utils import load_processed_ids, append_jsonl
    except ImportError:
        # Fallback if utils not found in path
        sys.path.append(str(Path(__file__).parent))
        from utils import load_processed_ids, append_jsonl

    processed_ids = load_processed_ids(jsonl_path, "image_id")
    
    # Filter out processed images
    # We need to map img_path to image_id to check against processed_ids
    # Logic from below: fileno = int(filename.stem)
    filtered_paths = []
    for p in img_paths:
        try:
            pid = int(p.stem)
            if pid not in processed_ids:
                filtered_paths.append(p)
        except ValueError:
            filtered_paths.append(p) # Keep if ID parsing fails, process it
            
    if len(filtered_paths) < len(img_paths):
        print(f"Resuming: Skipping {len(img_paths) - len(filtered_paths)} already processed images.")
        img_paths = filtered_paths

    new_data = []
    skipped = []
    
    for img_path in tqdm(img_paths, desc="Segmenting", unit="img"):
        try:
            # Load image to get dimensions
            img = Image.open(img_path).convert("RGB")
            height, width = img.height, img.width
            
            # Run segmentation
            masks, labels, vis_path = segmentor.segment(str(img_path))
            
            # Process results
            filename = img_path
            fileno = int(filename.stem)
            
            seg_id_scale = MAX_SEG_ID // len(labels) if len(labels) > 0 else 1
            out_mask = id2rgb(masks * seg_id_scale)
            pan_seg_file_name = f"seg_{filename.stem}.png"
            seg_img = Image.fromarray(out_mask)
            seg_img.save(out_dir / pan_seg_file_name)
            
            # Convert labels to category IDs
            class_labels = psg_metadata["thing_classes"] + psg_metadata["stuff_classes"]
            
            out_seg_info = []
            out_boxes = []
            
            for seg_idx, label in enumerate(labels):
                seg_id = (seg_idx + 1) * seg_id_scale
                bin_mask = torch.from_numpy(masks == (seg_idx + 1))
                
                # Find category ID for this label
                try:
                    category_id = class_labels.index(label)
                except ValueError:
                    # If label not found, skip this segment
                    continue
                
                area = bin_mask.sum().item()
                if area == 0:
                    continue
                
                out_seg_info.append(
                    dict(
                        id=seg_id,
                        category_id=category_id,
                        iscrowd=0,
                        isthing=1 if category_id < 80 else 0,
                        area=area,
                        score=1.0,  # kmax doesn't provide per-segment scores
                    )
                )
                out_boxes.append(
                    dict(
                        bbox=mask_to_boxes(bin_mask).tolist(),
                        bbox_mode=0,
                        category_id=category_id,
                        score=1.0,
                    )
                )
            
            if len(out_seg_info) == 0:
                skipped.append(str(img_path))
                continue
            
            # Make file_name relative to img_dir
            try:
                relative_path = img_path.relative_to(img_dir_path)
            except ValueError:
                # If path is not relative, use just the filename
                relative_path = img_path.name
            
            anno_entry = dict(
                file_name=str(relative_path),
                height=height,
                width=width,
                image_id=fileno,
                pan_seg_file_name=pan_seg_file_name,
                segments_info=out_seg_info,
                annotations=out_boxes,
            )
            
            new_data.append(anno_entry)
            append_jsonl(jsonl_path, anno_entry)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped.append(str(img_path))
    
    # Reconstruct full data for anno.json (merging new + old)
    # Read everything from jsonl to ensure consistent state
    print("re-reading full jsonl for anno.json...")
    full_data = [] 
    with open(jsonl_path, 'r') as f:
         for line in f:
             full_data.append(json.loads(line))

    anno_path = out_dir / "anno.json"
    with open(anno_path, "w") as f:
        json.dump(dict(data=full_data, skipped=skipped, **psg_metadata), f)
    
    print(f"Processed {len(new_data)} newly, Total {len(full_data)}")
    print(f"Annotations saved to: {anno_path}")
    return anno_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks using kMaX-DeepLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--img-dir", required=True, help="Directory containing input images (*.jpg or *.png)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--psg-meta", required=True, help="Path to psg_meta.json")
    parser.add_argument("--kmax-path", required=True, help="Path to kMaX-DeepLab repository")
    parser.add_argument("--kmax-config", required=True, help="Path to kMaX config file")
    parser.add_argument("--kmax-weights", required=True, help="Path to kMaX weights")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process (default: all)")
    
    args = parser.parse_args()
    
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    psg_meta = Path(args.psg_meta)
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        return 1
    
    if not psg_meta.exists():
        print(f"Error: PSG metadata not found: {psg_meta}")
        return 1
    
    if not Path(args.kmax_path).exists():
        print(f"Error: kMaX path not found: {args.kmax_path}")
        return 1
    
    if not Path(args.kmax_config).exists():
        print(f"Error: kMaX config not found: {args.kmax_config}")
        return 1
    
    if not Path(args.kmax_weights).exists():
        print(f"Error: kMaX weights not found: {args.kmax_weights}")
        return 1
    
    kmax_config = {
        'kmax_path': args.kmax_path,
        'config_file': args.kmax_config,
        'weights': args.kmax_weights,
        'output_dir': str(out_dir / 'kmax_vis'),
        'use_cuda': True,
    }
    
    generate_segmasks_kmax(
        img_dir=str(img_dir),
        out_dir=str(out_dir),
        psg_path=str(psg_meta),
        kmax_config=kmax_config,
        max_images=args.max_images,
    )
    
    print("\n=== Segmentation Complete ===")
    print(f"Outputs: {out_dir.absolute()}")
    print(f"\nYou can now use these outputs with generate_sg.py:")
    print(f"python generate_sg.py --skip-segmentation --anno-path {out_dir / 'anno.json'} ...")
    return 0


if __name__ == "__main__":
    exit(main())

