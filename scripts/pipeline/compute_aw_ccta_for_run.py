#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compute AW-CCTA for all models with available VQA+IGE scored files.")
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory (e.g. pipeline/run_1000_coco_images)")
    parser.add_argument("--vqa-dir", required=True, help="Directory containing per-model VQA scored files")
    parser.add_argument("--threshold", type=float, default=0.6, help="Quadrant threshold tau (default: 0.6)")
    parser.add_argument("--suffix", default="", help="Optional IGE suffix, e.g. '_equal' for ige_scored_<model>_equal.jsonl")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    scorer = root / "pipeline" / "scripts" / "consistency_agreement.py"
    final_graphs = Path(args.run_dir) / "final_graphs_pt"
    vqa_dir = Path(args.vqa_dir)

    models = sorted([p.name for p in final_graphs.iterdir() if p.is_dir()])
    if not models:
        print(f"No model directories found in {final_graphs}")
        return

    computed = 0
    skipped = 0
    for model in models:
        vqa_scored = vqa_dir / model / f"scored_{model}.jsonl"
        ige_scored = final_graphs / model / f"ige_scored_{model}{args.suffix}.jsonl"
        output = final_graphs / model / f"aw_ccta_{model}{args.suffix}.json"

        if not vqa_scored.exists() or not ige_scored.exists():
            skipped += 1
            print(f"[skip] {model} (missing VQA or IGE scored)")
            continue

        cmd = [
            "python", str(scorer),
            "--vqa-scored", str(vqa_scored),
            "--ige-scored", str(ige_scored),
            "--threshold", str(args.threshold),
            "--output", str(output),
        ]
        print(f"[run ] {model}")
        subprocess.run(cmd, check=True)
        computed += 1

    print("\nDone")
    print(f"  computed: {computed}")
    print(f"  skipped : {skipped}")


if __name__ == "__main__":
    main()
