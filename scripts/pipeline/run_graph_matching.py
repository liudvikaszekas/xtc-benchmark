#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def has_scene_graph_jsons(directory: Path) -> bool:
    return bool(list(directory.glob("scene-graph-description_*.json")) or list(directory.glob("scene-graph_*.json")))


def resolve_graph_dir(directory: str, role: str) -> Path:
    """
    Resolve an input graph directory from either:
    - a directory containing scene graph jsons directly
    - a parent directory with exactly one subdirectory containing scene graph jsons
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"{role} directory not found: {path}")

    if has_scene_graph_jsons(path):
        return path

    candidates = [p for p in sorted(path.iterdir()) if p.is_dir() and has_scene_graph_jsons(p)]
    if len(candidates) == 1:
        print(f"Resolved {role} directory to nested folder: {candidates[0]}")
        return candidates[0]

    if len(candidates) > 1:
        candidate_str = ", ".join(str(c) for c in candidates)
        raise ValueError(
            f"{role} directory {path} contains multiple candidate subdirectories with scene graphs: {candidate_str}. "
            "Pass the exact one via CLI."
        )

    raise ValueError(
        f"No scene graph JSON files found in {path}. Expected scene-graph_*.json or scene-graph-description_*.json"
    )

def run_command(cmd, desc):
    print(f"Running {desc}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in {desc}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Graph Matching Pipeline")
    parser.add_argument("--gt-sg-dir", required=True, help="Ground Truth Scene Graphs Dir")
    parser.add_argument("--gt-attr-file", required=True, help="Ground Truth Attributes JSON File")
    parser.add_argument("--pred-sg-dir", required=True, help="Prediction Scene Graphs Dir")
    parser.add_argument("--pred-attr-file", required=True, help="Prediction Attributes JSON File")
    parser.add_argument("--out-dir", required=True, help="Output Directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--alpha", type=float, default=0.7, help="Node similarity weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Edge similarity weight")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py_exec = sys.executable

    gt_sg_dir = resolve_graph_dir(args.gt_sg_dir, "GT SG")
    pred_sg_dir = resolve_graph_dir(args.pred_sg_dir, "Pred SG")
    
    # 1. Inject Attributes GT
    injected_gt_dir = out_dir / "injected_gt"
    cmd_inject_gt = [
        py_exec, str(script_dir / "inject_attributes.py"),
        "--sg-dir", str(gt_sg_dir),
        "--attr-file", str(args.gt_attr_file),
        "--out-dir", str(injected_gt_dir)
    ]
    run_command(cmd_inject_gt, "GT Attribute Injection")

    # 2. Inject Attributes Pred
    injected_pred_dir = out_dir / "injected_pred"
    cmd_inject_pred = [
        py_exec, str(script_dir / "inject_attributes.py"),
        "--sg-dir", str(pred_sg_dir),
        "--attr-file", str(args.pred_attr_file),
        "--out-dir", str(injected_pred_dir)
    ]
    run_command(cmd_inject_pred, "Pred Attribute Injection")
    
    # 3. Convert to matching format
    matching_input = out_dir / "scene_graphs_for_matching.json"
    cmd_convert = [
        py_exec, str(script_dir / "convert_to_matching_format.py"),
        "--gt-dir", str(injected_gt_dir),
        "--pt-dir", str(injected_pred_dir),
        "--output", str(matching_input)
    ]
    run_command(cmd_convert, "Format Conversion")
    
    # 2. Run Matching (Hungarian)
    # The script pipeline/scripts/match_graphs.py is match_openpsg_graphs.py
    matched_output = out_dir / "scene_graphs_matched.json"
    cmd_match = [
        py_exec, str(script_dir / "match_graphs.py"),
        "--input", str(matching_input),
        "--output", str(matched_output),
        "--model", args.model,
        "--high-recall", # Retained for compatibility with existing config
        "--strict-labels",
        "--alpha", str(args.alpha),
        "--beta", str(args.beta)
    ]
    run_command(cmd_match, "Graph Matching")
    
    # 3. Extract Prediction Masks (Optional but good for visualization)
    # This requires scene-graph.pkl in pred_dir?
    # If using CleanAndRefine, that output expects scene-graphs.
    # Prediction workflow runs: ImageGen -> Inference (generate_sg.py) -> Clean? -> Match.
    # Inference (generate_sg.py) produces scene-graph.pkl in its output dir.
    # If Clean ran, it produces JSONs.
    # match_graphs uses JSONs.
    # extract_pred_masks_from_scenegraph.py uses pkl?
    # Let's check extract_pred_masks args in workflow logic.
    # It passes --scene-graphs-dir.
    # If CleanAndRefine ran, does it produce a PKL?
    # clean_and_refine.py produces JSONs. It reads PKL.
    # Thus, if we match against Cleaned output, we might not have a PKL corresponding to cleaned output (unless clean script writes one).
    # clean_and_refine.py does NOT write PKL.
    # However, workflow logic says:
    # "if sg_pt_dir and (sg_pt_dir / 'scene-graph.pkl').exists():"
    # It looks for PKL in the PT dir. 
    # If PT dir is the *output* of cleaning, it won't have PKL.
    # But often PT dir is the output of inference (which has PKL).
    # IF matching uses Cleaned output, then Visualization might fail if it depends on PKL.
    # We will assume best effort.
    
    pkl_candidate = pred_sg_dir / "scene-graph.pkl"
    if pkl_candidate.exists():
        masks_out = out_dir / "prediction_masks_from_scenegraph.pkl"
        cmd_masks = [
            py_exec, str(script_dir / "extract_pred_masks_from_scenegraph.py"),
            "--scene-graphs-dir", str(pred_sg_dir),
            "--output", str(masks_out)
        ]
        # check if extract_pred_masks takes directory or pkl
        # Argument is --scene-graphs-dir.
        # run_command(cmd_masks, "Mask Extraction")
        # I'll invoke it but wrap in try/print because it's visualization aux.
        print("Running Mask Extraction...")
        subprocess.run(cmd_masks) # Don't exit on fail
    else:
        print(f"Warning: No scene-graph.pkl found in {pred_sg_dir}, skipping mask extraction.")

if __name__ == "__main__":
    main()
