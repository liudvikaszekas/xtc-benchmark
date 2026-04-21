import json
from pathlib import Path
from typing import Dict, Any, Set

def load_processed_ids(jsonl_path: Path, id_key: str = "image_id") -> Set[Any]:
    processed = set()
    if not jsonl_path.exists():
        return processed
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data.get(id_key))
            except:
                pass
    print(f"Loaded {len(processed)} processed IDs from {jsonl_path}")
    return processed

def append_jsonl(jsonl_path: Path, data: Dict[str, Any]):
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(data) + '\n')
