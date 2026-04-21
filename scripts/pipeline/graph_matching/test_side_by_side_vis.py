"""
Quick test script to generate side-by-side visualizations for a few images.
"""

import subprocess
import sys

# Example command
cmd = [
    sys.executable,
    "visualize_matched_masks.py",
    "--scene-graphs", "openpsg_graphs/openpsg_scene_graphs_test.json",
    "--matching-results", "openpsg_graphs/matching_results_test.json",
    "--pred-masks", "openpsg_graphs/openpsg_masks_test.pkl",
    "--output-dir", "./side_by_side_vis_test",
    "--max-images", "5",
    "--vis-mode", "side-by-side"
]

print("Running visualization command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, cwd="/home/mpws25/graph_matching")
sys.exit(result.returncode)
