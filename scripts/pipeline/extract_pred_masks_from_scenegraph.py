#!/usr/bin/env python3
"""
Create prediction masks pickle from `scene-graph.pkl` and scene-graph JSON files.

Output format matches expected `visualize_matched_masks.py` input:
{
  '<image_id>': {
      'masks': np.array of shape (N, H, W),
      'node_ids': [ 'node_0', 'node_1', ... ],
      'labels': [label1, label2, ...]
  },
  ...
}
"""
import pickle
import json
from pathlib import Path
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-graphs-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_graphs_dir)
    pkl_path = scene_dir / 'scene-graph.pkl'
    if not pkl_path.exists():
        print(f"scene-graph.pkl not found: {pkl_path}")
        return 1

    # Load pickle
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    # Load json files
    json_files = list(scene_dir.glob('scene-graph_*.json'))
    json_by_id = {}
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            json_by_id[str(data['image_id'])] = data

    out = {}
    for entry in pkl_data:
        img_id = str(entry.get('img_id'))
        if img_id not in json_by_id:
            continue
        json_entry = json_by_id[img_id]
        boxes = json_entry.get('boxes', [])
        mask = entry.get('mask')
        if mask is None:
            continue

        H, W = mask.shape
        masks_list = []
        node_ids = []
        labels = []
        for box in boxes:
            idx = box['index']
            node_ids.append(f"node_{idx}")
            labels.append(box.get('label', ''))
            bin_mask = (mask == idx).astype(np.uint8)
            masks_list.append(bin_mask)

        if masks_list:
            masks_arr = np.stack(masks_list, axis=0)
            out[img_id] = {
                'masks': masks_arr,
                'node_ids': node_ids,
                'labels': labels
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)

    print(f"Saved prediction masks to: {out_path} ({len(out)} images)")
    return 0


if __name__ == '__main__':
    exit(main())
