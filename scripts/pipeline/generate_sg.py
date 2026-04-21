#!/usr/bin/env python3
"""
Scene Graph Generation Pipeline

Workflow:
1. Generate segmentation masks from images
2. Convert masks to PSG format
3. Run scene graph inference
4. Visualize and export results
"""

import json
import pickle
import argparse
from pathlib import Path
from itertools import islice

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


MAX_SEG_ID = 256**3 - 1


class ImageDataset(Dataset):
    def __init__(self, img_dir, processor):
        img_path = Path(img_dir)
        self.paths = sorted(list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")) + list(img_path.glob("*.jpeg")))
        self.processor = processor
        self.processor_kwargs = {"return_tensors": "pt"}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        inputs = self.processor(images=img, **self.processor_kwargs)
        inputs["pixel_values"] = inputs["pixel_values"][0]
        inputs["pixel_mask"] = inputs["pixel_mask"][0]
        return inputs, img.height, img.width, str(p)


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


def process_segmentation(result, filename, height, width):
    filename = Path(filename)
    # Try to parse as int, otherwise use hash of filename
    try:
        fileno = int(filename.stem)
    except ValueError:
        fileno = abs(hash(filename.stem)) % (10**8)

    seg_mask = result["segmentation"].cpu()
    seg_info = result["segments_info"]
    
    if len(seg_info) == 0:
        return None, None

    seg_id_scale = MAX_SEG_ID // len(seg_info)
    out_mask = id2rgb(seg_mask.numpy() * seg_id_scale)
    pan_seg_file_name = f"seg_{filename.stem}.png"
    seg_img = Image.fromarray(out_mask)

    out_seg_info = []
    out_boxes = []
    for info in seg_info:
        seg_id = info["id"] * seg_id_scale
        bin_mask = seg_mask == info["id"]
        out_seg_info.append(
            dict(
                id=seg_id,
                category_id=info["label_id"],
                iscrowd=0,
                isthing=1 if info["label_id"] < 80 else 0,
                area=bin_mask.sum().item(),
                score=info["score"],
            )
        )
        out_boxes.append(
            dict(
                bbox=mask_to_boxes(bin_mask).tolist(),
                bbox_mode=0,
                category_id=info["label_id"],
                score=info["score"],
            )
        )

    return (
        dict(
            file_name=f"{filename.parent.name}/{filename.name}",
            height=height,
            width=width,
            image_id=fileno,
            pan_seg_file_name=pan_seg_file_name,
            segments_info=out_seg_info,
            annotations=out_boxes,
        ),
        seg_img,
    )


def generate_segmasks(img_dir, out_dir, psg_path, model_name, num_workers):
    print("\n=== Generating Segmentation Masks ===")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with open(psg_path) as f:
        psg_metadata = {
            k: v
            for k, v in json.load(f).items()
            if k in ("thing_classes", "stuff_classes", "predicate_classes")
        }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name, use_safetensors=True)
    model.to(device)
    print(f"Loaded model: {type(model).__name__}")

    class_labels = psg_metadata["thing_classes"] + psg_metadata["stuff_classes"]
    assert len(model.config.id2label) == len(class_labels)

    loader = DataLoader(
        dataset=ImageDataset(img_dir, processor),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    # JSONL Setup
    jsonl_path = out_dir / "segmentation.jsonl"
    try:
        from utils import load_processed_ids, append_jsonl
    except ImportError:
        sys.path.append(str(Path(__file__).parent))
        from utils import load_processed_ids, append_jsonl

    processed_ids = load_processed_ids(jsonl_path, "image_id")
    # Note: DataLoader doesn't easily support index skipping without rebuilding dataset.
    # We will skip inside loop for simplicity, but for efficiency we should filter dataset paths.
    
    # Filter dataset paths (accessing private attribs for hack fix)
    # dataset.paths is list of paths.
    orig_len = len(loader.dataset.paths)
    filtered_paths = []
    for p in loader.dataset.paths:
        try:
             pid = int(p.stem)
             if pid not in processed_ids:
                 filtered_paths.append(p)
        except:
             filtered_paths.append(p)
             
    loader.dataset.paths = filtered_paths # Hacky but cleaner than skipping loop
    
    print(f"Resuming: Processing {len(loader.dataset)} images (Skipped {orig_len - len(loader.dataset)})")

    new_data = []
    skipped = []
    
    with torch.no_grad():
        for inputs, height, width, abs_path in tqdm(loader, desc="Segmenting", unit="img"):
            height = height[0].item()
            width = width[0].item()
            abs_path = abs_path[0]

            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            outputs = model(**inputs)

            result = processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[(height, width)], label_ids_to_fuse=set()
            )[0]

            anno, seg_img = process_segmentation(result, abs_path, height, width)
            
            if anno is None:
                skipped.append(abs_path)
                continue

            seg_img.save(out_dir / anno["pan_seg_file_name"])
            new_data.append(anno)
            append_jsonl(jsonl_path, anno)
            
    # Re-read for full anno.json
    full_data = []
    with open(jsonl_path, 'r') as f:
         for line in f:
             full_data.append(json.loads(line))
             
    anno_path = out_dir / "anno.json"
    with open(anno_path, "w") as f:
        json.dump(dict(data=full_data, skipped=skipped, **psg_metadata), f)
    
    print(f"Processed {len(new_data)} images, skipped {len(skipped)}")
    print(f"Annotations saved to: {anno_path}")
    return anno_path


def convert_to_psg(anno_path, out_dir, min_segments=2):
    print("\n=== Converting to PSG Format ===")
    
    anno_path = Path(anno_path)
    out_dir = Path(out_dir)
    
    data = json.loads(anno_path.read_text())
    
    # Filter out images with fewer than min_segments (they cannot have relations)
    original_count = len(data["data"])
    filtered_data = []
    skipped_images = []
    
    for entry in data["data"]:
        num_segments = len(entry.get("segments_info", []))
        if num_segments >= min_segments:
            filtered_data.append(entry)
        else:
            skipped_images.append(entry)
    
    if skipped_images:
        print(f"\nFiltered out {len(skipped_images)} image(s) with < {min_segments} segments:")
        for skip in skipped_images:
            print(f"  - {skip['file_name']} (ID: {skip['image_id']}, segments: {len(skip.get('segments_info', []))})")
    
    image_ids = [entry["image_id"] for entry in filtered_data]

    psg_json = {
        "thing_classes": data["thing_classes"],
        "stuff_classes": data["stuff_classes"],
        "predicate_classes": data["predicate_classes"],
        "data": data["data"],  # Keep ALL images so downstream scripts can resolve metadata
        "test_image_ids": image_ids,
        "val_image_ids": [],
        "train_image_ids": [],
    }
    
    # Save as JSON (Required for inference)
    psg_path_json = out_dir / "custom_psg.json"
    psg_path_json.write_text(json.dumps(psg_json, indent=2))
    
    # Save as JSONL (User requested)
    psg_path_jsonl = out_dir / "custom_psg.jsonl"
    with open(psg_path_jsonl, 'w') as f:
        for entry in data["data"]:
            json.dump(entry, f)
            f.write('\n')

    # Also save metadata
    meta = {k: v for k, v in psg_json.items() if k != 'data'}
    (out_dir / "psg_metadata_generated.json").write_text(json.dumps(meta, indent=2))
    
    print(f"PSG format saved to: {psg_path_json} (and .jsonl)")
    print(f"Total images: {len(filtered_data)} (filtered from {original_count})")
    return psg_path_json, skipped_images


def check_inference_available():
    """Check if fair_psgg is available for inference."""
    try:
        import fair_psgg
        return True
    except ImportError:
        return False


def run_inference(model_dir, psg_path, img_root, seg_dir, output_path):
    print("\n=== Running Scene Graph Inference ===")
    
    # MUST import fair_psgg - no fallback, fail if not available
    from fair_psgg.tasks.inference import inference2
    
    inference2(
        model_folder=model_dir,
        output_path=output_path,
        anno_path=psg_path,
        img_dir=img_root,
        seg_dir=seg_dir,
        batch_size=4,
        num_workers=0,
        split="test",
        apply_sigmoid=True,
    )
    print(f"Inference results saved to: {output_path}")
    return output_path


def visualize_results(scene_path, psg_path, img_root, out_dir, top_k=15):
    print("\n=== Visualizing Results ===")
    
    scene_path = Path(scene_path)
    psg_path = Path(psg_path)
    img_root = Path(img_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with scene_path.open("rb") as f:
        scene_entries = pickle.load(f)
    
    with psg_path.open() as f:
        psg_meta = json.load(f)
    
    meta_by_img = {int(item["image_id"]): item for item in psg_meta["data"]}
    label_names = psg_meta["thing_classes"] + psg_meta["stuff_classes"]
    predicates = psg_meta["predicate_classes"]

    for entry in scene_entries:
        img_id = int(entry["img_id"])
        if img_id not in meta_by_img:
            # Image was filtered out of custom_psg prior to inference (e.g. < 2 segments)
            continue
            
        meta_entry = meta_by_img[img_id]
        img_path = (img_root / meta_entry["file_name"]).resolve()
        
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        fig_w = img.width / 100
        fig_h = img.height / 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax.imshow(img)
        ax.axis("off")

        boxes_json = []
        for idx, (bbox, label_idx) in enumerate(zip(entry["bboxes"], entry["box_label"])):
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), w, h, fill=False, edgecolor="lime", linewidth=2)
            ax.add_patch(rect)
            ax.text(
                x1, max(y1 - 5, 5),
                f"{idx}: {label_names[label_idx]}",
                color="lime", fontsize=10, backgroundcolor="black",
            )
            boxes_json.append({
                "index": idx,
                "label": label_names[label_idx],
                "bbox_xyxy": [x1, y1, x2, y2],
            })

        relations = []
        for (sbj_idx, obj_idx), scores in zip(entry["pairs"], entry["rel_scores"]):
            sbj_idx = int(sbj_idx)
            obj_idx = int(obj_idx)
            best_rel_idx = int(np.argmax(scores[1:])) + 1
            relations.append({
                "subject_index": sbj_idx,
                "subject_label": label_names[entry["box_label"][sbj_idx]],
                "object_index": obj_idx,
                "object_label": label_names[entry["box_label"][obj_idx]],
                "predicate": predicates[best_rel_idx - 1],
                "predicate_index": best_rel_idx - 1,
                "predicate_score": float(scores[best_rel_idx]),
                "no_relation_score": float(scores[0]),
            })

        top_relations = list(
            islice(
                sorted(relations, key=lambda r: r["predicate_score"], reverse=True),
                top_k,
            )
        )

        out_stem = out_dir / f"scene-graph_{img_id}"
        fig.tight_layout()
        fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        out_json = {
            "image_id": img_id,
            "file_name": meta_entry["file_name"],
            "boxes": boxes_json,
            "relations": relations,
            "top_relations": top_relations,
        }
        out_stem.with_suffix(".json").write_text(json.dumps(out_json, indent=2))
        
        # Append to JSONL
        try:
            from utils import append_jsonl
            append_jsonl(out_dir / "scene_graphs.jsonl", out_json)
        except:
             pass
             
        print(f"Visualized: {out_stem.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Scene graph generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--img-dir", required=True, help="Directory containing input images")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--psg-meta", required=True, help="Path to psg_meta.json")
    parser.add_argument("--model-dir", help="Path to trained model (optional)")
    parser.add_argument(
        "--seg-model",
        default="facebook/mask2former-swin-large-coco-panoptic",
        help="Segmentation model",
    )
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--top-k", type=int, default=15, help="Top K relations")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference")
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip segmentation (use existing anno.json)")
    parser.add_argument("--anno-path", help="Path to existing anno.json (required if --skip-segmentation)")
    
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
    
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return 1
        if not (model_dir / "config.json").exists():
            print(f"Error: config.json not found in {model_dir}")
            return 1
    
    # Validate skip-segmentation requirements
    if args.skip_segmentation:
        if not args.anno_path:
            print("Error: --anno-path is required when using --skip-segmentation")
            return 1
        anno_path = Path(args.anno_path)
        if not anno_path.exists():
            print(f"Error: Annotation file not found: {anno_path}")
            return 1
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate or use existing segmentation
    if args.skip_segmentation:
        print("\n=== Using Existing Segmentation ===")
        anno_path = Path(args.anno_path)
        print(f"Loading annotations from: {anno_path}")
        seg_dir = anno_path.parent
    else:
        anno_path = generate_segmasks(
            img_dir=str(img_dir),
            out_dir=str(out_dir),
            psg_path=str(psg_meta),
            model_name=args.seg_model,
            num_workers=args.workers,
        )
        seg_dir = out_dir
    
    psg_path, skipped_entries = convert_to_psg(anno_path, out_dir)
    
    if not args.skip_inference and args.model_dir:
        scene_graph_path = out_dir / "scene-graph.pkl"
        result = run_inference(
            model_dir=args.model_dir,
            psg_path=psg_path,
            img_root=img_dir,
            seg_dir=seg_dir,
            output_path=scene_graph_path,
        )
        
        if result:
            if skipped_entries:
                print(f"Adding empty relation records for {len(skipped_entries)} skipped images...")
                with open(scene_graph_path, 'rb') as f:
                    scene_entries = pickle.load(f)
                    
                for entry in skipped_entries:
                    img_id = entry["image_id"]
                    bboxes = []
                    box_labels = []
                    
                    for ann, seg in zip(entry.get("annotations", []), entry.get("segments_info", [])):
                        x, y, w, h = ann["bbox"]
                        bboxes.append([x, y, x + w, y + h])
                        box_labels.append(seg["category_id"])
                        
                    scene_entries.append({
                        "img_id": img_id,
                        "bboxes": np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
                        "box_label": np.array(box_labels, dtype=np.int64) if box_labels else np.zeros((0,), dtype=np.int64),
                        "pairs": np.zeros((0, 2), dtype=np.int64),
                        "rel_scores": np.zeros((0, 57), dtype=np.float32)  # 56 predicates + 1 for no_relation
                    })
                    
                with open(scene_graph_path, 'wb') as f:
                    pickle.dump(scene_entries, f)
                print("Successfully updated scene-graph.pkl with skipped image entries.")

            visualize_results(
                scene_path=scene_graph_path,
                psg_path=psg_path,
                img_root=img_dir,
                out_dir=out_dir,
                top_k=args.top_k,
            )
    elif args.skip_inference:
        print("\nSkipped inference (--skip-inference)")
    else:
        print("\nSkipped inference (no --model-dir)")
    
    print(f"\n=== Complete ===")
    print(f"Outputs: {out_dir.absolute()}")
    return 0


if __name__ == "__main__":
    main()

