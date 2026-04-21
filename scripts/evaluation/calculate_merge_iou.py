
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
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)

def process_image(image_data, padding, min_group_size):
    segments_info = image_data['segments_info']
    annotations = image_data['annotations']
    image_id = image_data['image_id']
    
    seg_to_bbox = {}
    for seg, ann in zip(segments_info, annotations):
        seg_to_bbox[seg['id']] = ann['bbox']
    
    category_groups = defaultdict(list)
    for seg in segments_info:
        category_groups[seg['category_id']].append(seg['id'])
    
    results = []
    
    for category_id, seg_ids in category_groups.items():
        if len(seg_ids) < min_group_size:
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
        
        for root, group_members in groups.items():
            if len(group_members) < min_group_size:
                continue
            
            # For each merged group, find the maximum pairwise IoU between any two members
            max_group_iou = -1.0
            best_pair = (None, None)
            
            for i, sid1 in enumerate(group_members):
                for sid2 in group_members[i+1:]:
                    iou = calculate_iou(padded_boxes[sid1], padded_boxes[sid2])
                    if iou > max_group_iou:
                        max_group_iou = iou
                        best_pair = (sid1, sid2)
            
            if max_group_iou >= 0:
                results.append({
                    'iou': max_group_iou,
                    'image_id': image_id,
                    'category_id': category_id,
                    'sid1': best_pair[0],
                    'sid2': best_pair[1],
                    'group_size': len(group_members)
                })
                
    return results

def main():
    anno_path = "/sc/home/liudvikas.zekas/vlm-benchmark/pipeline/run_1000_coco_images/1_segmentation_gt/anno.json"
    padding = 10
    min_group_size = 3
    
    with open(anno_path, 'r') as f:
        data = json.load(f)
    if 'data' not in data:
        print(f"Error: 'data' key not found in {anno_path}")
        return
    print(f"Processing {len(data['data'])} images...")
    
    all_results = []
    for image_data in data['data']:
        all_results.extend(process_image(image_data, padding, min_group_size))
    
    if not all_results:
        print("No merges found.")
        return
    
    all_ious = [r['iou'] for r in all_results]
    min_iou = min(all_ious)
    mean_iou = sum(all_ious) / len(all_ious)
    
    print(f"Total merged groups processed: {len(all_ious)}")
    print(f"Minimum Max-Pairwise IoU: {min_iou:.6f}")
    print(f"Mean Max-Pairwise IoU: {mean_iou:.6f}")
    
    print("\nTop 10 Minimum Max-Pairwise IoUs (per merged group):")
    all_results.sort(key=lambda x: x['iou'])
    for i, res in enumerate(all_results[:10]):
        print(f"{i+1}. IoU: {res['iou']:.6f} (Image: {res['image_id']}, Cat: {res['category_id']}, Size: {res['group_size']}, Segs: {res['sid1']}, {res['sid2']})")

if __name__ == "__main__":
    main()
