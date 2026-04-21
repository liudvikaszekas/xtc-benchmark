#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Prediction Scene Graph Inference")
    parser.add_argument("--model", required=True, help="Model name (e.g. showo2)")
    parser.add_argument("--img_dir", required=True, help="Input images dir")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--psg-meta", required=False, help="Path to psg_meta.json")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    psg_meta = args.psg_meta
    if not psg_meta:
        psg_meta_path = script_dir / "psg_meta.json"
        if psg_meta_path.exists():
            psg_meta = str(psg_meta_path)
        else:
            print("Warning: psg_meta.json not found and not provided!")
            # It might fail in generate_sg.py
            psg_meta = str(psg_meta_path) # Pass anyway to let generate_sg fail responsibly

    cmd = [
        "python", str(script_dir / "generate_sg.py"),
        "--img-dir", args.img_dir,
        "--out-dir", args.out_dir,
        "--psg-meta", psg_meta,
    ]
    
    print(f"Running inference wrapper for {args.model}...")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
