#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict


class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)


def merge_segments_for_image(image_data, padding, min_group_size):
    segments_info = image_data['segments_info']
    annotations = image_data['annotations']
    
    seg_to_bbox = {}
    for seg, ann in zip(segments_info, annotations):
        assert seg['category_id'] == ann['category_id']
        seg_to_bbox[seg['id']] = ann['bbox']
    
    category_groups = defaultdict(list)
    for seg in segments_info:
        category_groups[seg['category_id']].append(seg['id'])
    
    id_mapping = {}
    for category_id, seg_ids in category_groups.items():
        if len(seg_ids) == 1:
            id_mapping[seg_ids[0]] = seg_ids[0]
            continue
        
        padded_boxes = {}
        for seg_id in seg_ids:
            x1, y1, x2, y2 = seg_to_bbox[seg_id]
            padded_boxes[seg_id] = [x1 - padding, y1 - padding, x2 + padding, y2 + padding]
        
        uf = UnionFind(seg_ids)
        for i, seg_id1 in enumerate(seg_ids):
            for seg_id2 in seg_ids[i+1:]:
                if boxes_overlap(padded_boxes[seg_id1], padded_boxes[seg_id2]):
                    uf.union(seg_id1, seg_id2)
        
        groups = defaultdict(list)
        for seg_id in seg_ids:
            root = uf.find(seg_id)
            groups[root].append(seg_id)
        
        for group_members in groups.values():
            if len(group_members) >= min_group_size:
                merged_id = min(group_members)
                for seg_id in group_members:
                    id_mapping[seg_id] = merged_id
            else:
                # Small group - keep original IDs (no merge)
                for seg_id in group_members:
                    id_mapping[seg_id] = seg_id
    
    return id_mapping


def main():
    parser = argparse.ArgumentParser(description='Merge object segments based on overlapping bounding boxes')
    parser.add_argument('--input', required=True, help='Path to anno.json file')
    parser.add_argument('--padding', type=int, default=0, help='Padding to add to bounding boxes')
    parser.add_argument('--min-group-size', type=int, default=3, help='Minimum number of objects required to form a merged group')
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        anno_data = json.load(f)
    
    output_mapping = {}
    for image_data in anno_data['data']:
        image_id = str(image_data['image_id'])
        id_mapping = merge_segments_for_image(image_data, args.padding, args.min_group_size)
        output_mapping[image_id] = id_mapping
    
    output_file = args.input.replace('.json', '_merged.json')
    with open(output_file, 'w') as f:
        json.dump(output_mapping, f, indent=2)
    
    print(f"Merge mapping saved to: {output_file}")


if __name__ == '__main__':
    main()

