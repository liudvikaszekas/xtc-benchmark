import json
import argparse
from pathlib import Path

def inject(sg_dir, attr_json_path, out_dir):
    sg_dir = Path(sg_dir)
    attr_json_path = Path(attr_json_path)
    out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not attr_json_path.exists():
        print(f"Attr file {attr_json_path} does not exist, skipping.")
        return
        
    with open(attr_json_path, 'r') as f:
        attrs = json.load(f)
        
    attr_map_idx = {}
    attr_map_seg = {}
    for a in attrs:
        img_id = str(a['image_id'])
        if img_id not in attr_map_idx:
            attr_map_idx[img_id] = {}
        if img_id not in attr_map_seg:
            attr_map_seg[img_id] = {}

        data = a.get('attributes', {})
        if 'index' in a and a['index'] is not None:
            attr_map_idx[img_id][a['index']] = data
        if 'seg_id' in a and a['seg_id'] is not None:
            attr_map_seg[img_id][str(a['seg_id'])] = data
            
    for sg_file in sg_dir.glob('scene-graph*.json'):
        with open(sg_file, 'r') as f:
            sg = json.load(f)
            
        img_id = str(sg.get('image_id'))
        if img_id in attr_map_idx or img_id in attr_map_seg:
            for box in sg.get('boxes', []):
                idx = box.get('index')
                sid_main = str(box.get('id', ''))
                sids = [str(s) for s in box.get('seg_ids', [])]
                if not sids and sid_main:
                    sids = [sid_main]
                
                # 1. Direct index match
                if idx is not None and idx in attr_map_idx.get(img_id, {}):
                    box['attributes'] = attr_map_idx[img_id][idx]
                
                # 2. seg_ids match (handles merged nodes)
                if img_id in attr_map_seg:
                    img_attrs = attr_map_seg[img_id]
                    # If it is a group node, we might want to populate member_attributes
                    if len(sids) > 1:
                        if 'member_attributes' not in box:
                            box['member_attributes'] = []
                        for sid in sids:
                            if sid in img_attrs:
                                box['member_attributes'].append({
                                    'seg_id': sid,
                                    'attributes': img_attrs[sid]
                                })
                    
                    # Also try to populate main attributes if missing and matches sid_main
                    if not box.get('attributes') and sid_main in img_attrs:
                        box['attributes'] = img_attrs[sid_main]
                    
                    # Fallback: if main attributes still missing, pick first available from sids
                    if not box.get('attributes'):
                        for sid in sids:
                            if sid in img_attrs:
                                box['attributes'] = img_attrs[sid]
                                break
                    
        with open(out_dir / sg_file.name, 'w') as f:
            json.dump(sg, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sg-dir', required=True)
    parser.add_argument('--attr-file', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    inject(args.sg_dir, args.attr_file, args.out_dir)
