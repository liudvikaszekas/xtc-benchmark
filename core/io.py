import json
import os
from pathlib import Path
from typing import Any, Dict, List

def append_jsonl(path: Path, data: Dict[str, Any]):
    """Appends a single record to a JSONL file."""
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Reads all records from a JSONL file."""
    if not path.exists():
        return []
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def get_last_processed_index(path: Path, key: str = 'image_id') -> Any:
    """
    Returns the value of 'key' from the last record in the JSONL file.
    Returns None if file is empty or missing.
    """
    if not path.exists():
        return None
    
    last_line = None
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                last_line = line
                
    if last_line:
        try:
            return json.loads(last_line).get(key)
        except:
            return None
    return None
