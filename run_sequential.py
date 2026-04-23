#!/usr/bin/env python3
"""
XTC-Bench sequential pipeline runner (``benchmark/run_sequential.py``)

Runs the full evaluation pipeline on one machine, step by step. For cluster submission of 
the same logical pipeline, use ``benchmark/run_benchmark.py`` with a JSON config.

Command-line arguments (run ``python benchmark/run_sequential.py --help`` for the live list):

  Required (unless both are set in ``--config`` JSON as ``images`` and ``output_dir`` / ``run_dir``):

    --images DIR          Input images (e.g. COCO val).
    --output-dir DIR      Run directory; artifacts and ``run_config.json`` are written here.

  Data / scope:

    --gt-dataset DIR      Pre-computed GT bundle (required subdirs are symlinked into
                          ``--output-dir``). When set, steps **1–7 are never scheduled**; run
                          ``--steps`` in the ``8–19`` range (or a subset) for prediction and eval.
    --models M1 [M2 ...] Image-generation / evaluation model names (e.g. ``januspro showo2``).
    --steps RANGE         Comma ranges and single steps, e.g. ``1-7``, ``8-19``, ``1-5,8,14-19``.
                          Default when omitted: from config, else ``1-19``.

  Models and GPUs (each can be omitted to take value from ``--config``):

    --llm-clean-refine ID   HF model for steps 3 and 11 (relation clean/refine).
    --llm-attributes ID     HF model for steps 5 and 13 (attributes).
    --llm-prompt ID         HF model for step 6 (prompt refinement when enabled).
    --llm-judge ID          HF model for steps 16 and 19 (LLM judge).
    --num-gpus N            GPUs for VLM steps above (default 1 or from config).
    Judge GPU count comes from config key ``judge_num_gpus`` or ``evaluation.llm_judge.slurm.gpus``,
    else falls back to ``num_gpus``.

  JSON config:

    --config PATH         Pipeline-style JSON (same sections as ``run_full_pipeline`` /
                          ``pipeline_factory``) and/or flat keys. Any CLI flag you **omit** is
                          filled from this file. Any flag you **pass** overrides the file.

  Execution:

    --dry-run               Print each step’s shell command; do not run.
    --conda-init CMD        Shell snippet to initialise conda before ``conda activate`` (optional;
                            auto-detected from setup scripts when omitted).
    --refine-sentences      Step 6: enable LLM sentence refinement.
    --no-refine-objects     Step 6: disable per-object description refinement.
    --save-logs             Write each step’s stdout/stderr under ``<output-dir>/logs/``.

  Precedence: explicit CLI > JSON > built-in defaults (paths, LLM ids, conda env names, etc.).

Config file highlights (pipeline sections mirror Slurm configs):

  - Nested blocks: ``segmentation``, ``scene_graph_generation``, ``clean_and_refine_relations``,
    ``graph_merging``, ``attribute_generation``, ``prompt_generation``, ``evaluation``,
    ``image_generation``, ``graph_matching``. Each may set ``conda_env`` or ``slurm.conda_env``,
    ``model`` / ``llm.model``, paths (e.g. ``kmax_config``, ``psg_meta_path``), ``num_gpus`` /
    ``slurm.gpus``, graph merge ``padding`` / ``min_group_size`` / ``threshold``, etc.
    For ``evaluation`` only: a root ``conda_env`` applies to **step 15** (IGE), not to the LLM
    judge. Steps **16** and **19** use ``evaluation.llm_judge`` (``conda_env`` or
    ``slurm.conda_env``) when set; otherwise the default ``vllm_env`` because ``llm_judge.py``
    requires the ``vllm`` package.
  - Top-level or flat: ``images``, ``output_dir``, ``run_dir``, ``models``, ``steps``,
    ``gt_dataset``, ``external_gt_run_dir``, ``conda_init_script``, ``llm_*``, ``num_gpus``,
    ``judge_num_gpus``, ``psg_meta_path``, ``kmax_*``, ``vqa_*`` paths, ``refine_objects``,
    ``refine_sentences``, ``save_logs``, ``dry_run``.
  - Per-step conda overrides: ``step_conda_envs`` / ``conda_envs`` map, e.g. ``{"3": "my_env"}``.

Examples:

  # Full run from the repo root (GT through evaluation) for one image model:
  python benchmark/run_sequential.py \\
      --images /path/to/coco_val2017 \\
      --output-dir outputs/my_run \\
      --models januspro \\
      --save-logs

  # Drive almost everything from JSON; only paths/models on CLI where you like:
  python benchmark/run_sequential.py \\
      --config benchmark/configs/my_run.json \\
      --images benchmark/dataset \\
      --output-dir benchmark/test

  # Same JSON but override one LLM from the shell:
  python benchmark/run_sequential.py --config benchmark/configs/my_run.json \\
      --images benchmark/dataset --output-dir benchmark/test \\
      --llm-judge Qwen/Qwen2.5-7B-Instruct

  # Pre-staged GT under output-dir; run only prediction + eval steps:
  python benchmark/run_sequential.py \\
      --images /path/to/same_images_as_gt \\
      --output-dir /path/to/run_with_gt_symlinked \\
      --gt-dataset /path/to/precomputed_gt_bundle \\
      --models januspro \\
      --steps 8-19

  # Subset of steps (e.g. regenerate prompts only):
  python benchmark/run_sequential.py \\
      --images /path/to/images \\
      --output-dir outputs/partial \\
      --models showo2 \\
      --steps 6

  # Inspect commands without executing:
  python benchmark/run_sequential.py \\
      --images /path/to/images \\
      --output-dir outputs/dry \\
      --models januspro \\
      --dry-run

  # Custom conda bootstrap (when auto-detection is wrong):
  python benchmark/run_sequential.py \\
      --images /path/to/images \\
      --output-dir outputs/run \\
      --models januspro \\
      --conda-init 'source /path/to/miniconda3/etc/profile.d/conda.sh'

Pipeline steps (default conda env names are overridden by ``conda_env`` / ``step_conda_envs``
in config):

  --- Ground truth (steps 1–7) ---
  1. Segmentation (kMaX-DeepLab)
  2. Scene graph generation (fair-psg)
  3. Clean & refine relations (VLM validation)
  4. Graph merging (overlap-based segment merging)
  5. Attribute generation (VLM extraction)
  6. Prompt generation (scene graph → text prompt)
  7. Question generation (scene graph → VQA questions)

  --- Per-model prediction (steps 8–19, repeated for each ``--models`` entry) ---
  8.  Image generation (from prompts)
  9.  Segmentation (predicted images)
  10. Scene graph generation (predicted images)
  11. Clean & refine relations (predicted)
  12. Graph merging (predicted)
  13. Attribute generation (predicted)
  14. Graph matching (GT ↔ predicted, Hungarian)
  15. Scene graph eval / IGE (lookup answers from matched graphs)
  16. LLM judge — scene graphs (score IGE answers)
  17. VQA generation (VLM answers on generated images)
  18. VQA merge (chunked results)
  19. LLM judge — VQA (score VQA answers)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk up from this script to find the repo root (directory containing configs/)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        # In xtc-benchmark, configs/ and scripts/ are at the root
        if (current / "configs").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent
    # Fallback: assume script is at the repo root
    return Path(__file__).resolve().parent

REPO_ROOT = _find_repo_root()

# Default weight/config locations
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights"
DEFAULT_KMAX_SUBMODULE = REPO_ROOT / "submodules" / "kmax-deeplab"
DEFAULT_KMAX_CONFIG = REPO_ROOT / "scripts" / "pipeline" / "segmentation" / "config" / "kmax_convnext_large.yaml"
DEFAULT_KMAX_WEIGHTS = DEFAULT_WEIGHTS_DIR / "kmax_convnext_large.pth"
DEFAULT_PSG_MODEL_DIR = DEFAULT_WEIGHTS_DIR / "models" / "masks-loc-sem"
DEFAULT_PSG_META = REPO_ROOT / "configs" / "psg_metadata.json"
DEFAULT_CATEGORY_MAPPING = REPO_ROOT / "configs" / "updated_category_mapping.json"
DEFAULT_VQA_METADATA = REPO_ROOT / "configs" / "vqa_metadata.json"
DEFAULT_VQA_SYNONYMS = REPO_ROOT / "configs" / "vqa_synonyms.json"
DEFAULT_VQA_TEMPLATES = REPO_ROOT / "scripts" / "evaluation" / "question_generation" / "templates"

# Conda environment names (matching setup.sh / setup_v2.sh)
ENV_KMAX = "kmax_env"
ENV_FAIR_PSG = "fair-psg"
ENV_VLLM = "vllm_env"
ENV_GENERATE_ANY_SCENE = "GenerateAnyScene"
ENV_UNIVLM = "univlm"
ENV_GRAPH_MATCHING = "graph_matching"

# Step ID -> conda environment mapping
STEP_ENVS = {
    1:  ENV_KMAX,
    2:  ENV_FAIR_PSG,
    3:  ENV_VLLM,
    4:  ENV_FAIR_PSG,
    5:  ENV_VLLM,
    6:  ENV_VLLM,
    7:  ENV_GENERATE_ANY_SCENE,
    8:  ENV_UNIVLM,
    9:  ENV_KMAX,
    10: ENV_FAIR_PSG,
    11: ENV_VLLM,
    12: ENV_FAIR_PSG,
    13: ENV_VLLM,
    14: ENV_GRAPH_MATCHING,
    15: ENV_VLLM,  # image_generation_eval.py has no heavy deps, but runs in vllm for consistency
    16: ENV_VLLM,
    17: ENV_UNIVLM,
    18: ENV_UNIVLM,
    19: ENV_VLLM,
}


def _conda_env_from_step_block(block: Any) -> Optional[str]:
    """Read conda env name from a pipeline step dict (top-level or under slurm)."""
    if not isinstance(block, dict):
        return None
    env = block.get("conda_env")
    if env:
        s = str(env).strip()
        return s or None
    slurm = block.get("slurm") or {}
    env = slurm.get("conda_env")
    if env:
        s = str(env).strip()
        return s or None
    return None


def collect_step_conda_envs_from_cfg(cfg: Dict[str, Any]) -> Dict[int, str]:
    """
    Build per-step-id conda env overrides from the same JSON shape as the Slurm
    pipeline (step sections + optional explicit map). Flat ``step_conda_envs``
    / ``conda_envs`` wins last (keys are step ids as int or string).

    ``evaluation`` root ``conda_env`` applies only to step 15; steps 16 and 19 read
    ``evaluation.llm_judge`` so a UnivLM-style root env does not break ``llm_judge.py``.
    """
    if not isinstance(cfg, dict):
        return {}
    acc: Dict[int, str] = {}

    section_to_steps = (
        ("segmentation", (1, 9)),
        ("scene_graph_generation", (2, 10)),
        ("clean_and_refine_relations", (3, 11)),
        ("graph_merging", (4, 12)),
        ("attribute_generation", (5, 13)),
        ("prompt_generation", (6,)),
        ("image_generation", (8,)),
        ("graph_matching", (14,)),
    )
    for section, step_ids in section_to_steps:
        env = _conda_env_from_step_block(cfg.get(section))
        if env:
            for sid in step_ids:
                acc[sid] = env

    ev = cfg.get("evaluation")
    if isinstance(ev, dict):
        # llm_judge.py imports vLLM — steps 16 / 19 must not inherit a generic evaluation
        # conda env (often "univlm" for image/VQA steps). Use llm_judge-specific conda only.
        judge_conda = _conda_env_from_step_block(ev.get("llm_judge"))
        if judge_conda:
            acc[16] = judge_conda
            acc[19] = judge_conda
        # Root evaluation conda (if any) applies to step 15 (IGE) only, not the judge.
        ev_conda = _conda_env_from_step_block(ev)
        if ev_conda:
            acc.setdefault(15, ev_conda)
        qg_conda = _conda_env_from_step_block(ev.get("question_generation"))
        if qg_conda:
            acc.setdefault(7, qg_conda)

    flat = cfg.get("step_conda_envs") or cfg.get("conda_envs")
    if isinstance(flat, dict):
        for k, v in flat.items():
            if v is not None and str(v).strip():
                acc[int(k)] = str(v).strip()
    return acc


# Directory naming convention for each step's output
DIR_NAMES = {
    1:  "1_segmentation_gt",
    2:  "2_scene_graphs_gt",
    3:  "3_clean_and_refine_gt",
    4:  "4_graph_merge_gt",
    5:  "5_attributes_gt",
    6:  "6_prompt_generation",
    7:  "vqa_questions",
    8:  "7_images",           # + /<model>
    9:  "8_segmentation_pt",  # + /<model>
    10: "9_scene_graphs_pt",  # + /<model>
    11: "10_clean_and_refine_pt",  # + /<model>
    12: "11_graph_merge_pt",       # + /<model>
    13: "12_attributes_pt",        # + /<model>
    14: "final_graphs_pt",         # + /<model>
    # 15-19 outputs go into final_graphs_pt/<model> or vqa_outputs/<model>
}

# Keys that indicate a pipeline JSON (same layout as run_full_pipeline / pipeline_factory).
_PIPELINE_SECTION_KEYS = frozenset(
    {
        "segmentation",
        "scene_graph_generation",
        "clean_and_refine_relations",
        "graph_merging",
        "attribute_generation",
        "prompt_generation",
        "evaluation",
        "image_generation",
        "graph_matching",
    }
)


def _is_pipeline_style_cfg(cfg: Dict[str, Any]) -> bool:
    if not isinstance(cfg, dict):
        return False
    return any(
        k in cfg and isinstance(cfg[k], dict) for k in _PIPELINE_SECTION_KEYS
    )


def _gpus_from_block(block: Optional[Dict[str, Any]]) -> Optional[int]:
    if not block:
        return None
    if block.get("num_gpus") is not None:
        return int(block["num_gpus"])
    slurm = block.get("slurm") or {}
    if slurm.get("gpus") is not None:
        return int(slurm["gpus"])
    return None


def extract_settings_from_json(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten pipeline-style JSON (nested step blocks) and/or legacy flat keys into
    knobs consumed by this script. Values here are defaults; explicit CLI flags
    still override in main().
    """
    out: Dict[str, Any] = {}

    def take(key: str, val: Any) -> None:
        if val is not None:
            out[key] = val

    # --- Legacy / top-level keys (always honored when present) ---
    take("llm_clean_refine", cfg.get("llm_clean_refine"))
    take("llm_attributes", cfg.get("llm_attributes"))
    take("llm_prompt", cfg.get("llm_prompt"))
    take("llm_judge", cfg.get("llm_judge"))
    if cfg.get("num_gpus") is not None:
        take("num_gpus", int(cfg["num_gpus"]))
    take("models", cfg.get("models"))
    take("steps", cfg.get("steps"))
    take("images", cfg.get("images"))
    take("output_dir", cfg.get("output_dir") or cfg.get("run_dir"))
    if cfg.get("gt_dataset") is not None:
        take("gt_dataset", cfg.get("gt_dataset"))
    elif cfg.get("external_gt_run_dir"):
        take("gt_dataset", cfg.get("external_gt_run_dir"))
    ci = cfg.get("conda_init_script") or cfg.get("conda_init")
    if ci:
        take("conda_init", ci)
    take("vqa_metadata_path", cfg.get("vqa_metadata") or cfg.get("vqa_metadata_path"))
    take("vqa_synonyms_path", cfg.get("vqa_synonyms") or cfg.get("vqa_synonyms_path"))
    take("vqa_templates_dir", cfg.get("vqa_templates_dir") or cfg.get("vqa_template_dir"))
    take("psg_meta_path", cfg.get("psg_meta_path"))
    take("kmax_path", cfg.get("kmax_path"))
    take("kmax_config_path", cfg.get("kmax_config_path"))
    take("kmax_weights_path", cfg.get("kmax_weights_path"))
    take("psg_model_dir", cfg.get("psg_model_dir"))
    take("category_mapping_path", cfg.get("category_mapping_path"))
    take("graph_matching_model", cfg.get("graph_matching_model"))
    if "graph_merge_padding" in cfg:
        out["graph_merge_padding"] = str(cfg["graph_merge_padding"])
    if "graph_merge_min_group_size" in cfg:
        out["graph_merge_min_group_size"] = str(cfg["graph_merge_min_group_size"])
    if "graph_merge_threshold" in cfg:
        out["graph_merge_threshold"] = str(cfg["graph_merge_threshold"])
    if "judge_num_gpus" in cfg and cfg["judge_num_gpus"] is not None:
        out["judge_num_gpus"] = int(cfg["judge_num_gpus"])
    if "save_logs" in cfg:
        out["save_logs"] = bool(cfg["save_logs"])
    if "dry_run" in cfg:
        out["dry_run"] = bool(cfg["dry_run"])
    if "refine_objects" in cfg:
        out["refine_objects"] = bool(cfg["refine_objects"])
    if "refine_sentences" in cfg:
        out["refine_sentences"] = bool(cfg["refine_sentences"])

    if not _is_pipeline_style_cfg(cfg):
        return out

    seg = cfg.get("segmentation") or {}
    sg = cfg.get("scene_graph_generation") or {}
    clean = cfg.get("clean_and_refine_relations") or {}
    merge = cfg.get("graph_merging") or {}
    attr = cfg.get("attribute_generation") or {}
    prompt = cfg.get("prompt_generation") or {}
    prompt_llm = prompt.get("llm") or {}
    judge = cfg.get("evaluation") or {}
    llm_judge = judge.get("llm_judge") or {}
    matching = cfg.get("graph_matching") or {}

    psg = clean.get("psg_meta_path") or seg.get("psg_meta_path") or sg.get("psg_meta_path")
    take("psg_meta_path", psg)

    kc = seg.get("kmax_config") or {}
    if isinstance(kc, dict):
        take("kmax_path", kc.get("path"))
        take("kmax_config_path", kc.get("config"))
        take("kmax_weights_path", kc.get("weights"))

    take("psg_model_dir", sg.get("model_dir"))
    take("category_mapping_path", attr.get("mapping_json"))

    take("llm_clean_refine", clean.get("model"))
    take("llm_attributes", attr.get("model"))
    take("llm_prompt", prompt_llm.get("model"))
    take("llm_judge", llm_judge.get("model"))

    if "refine_objects" in prompt:
        take("refine_objects", bool(prompt["refine_objects"]))
    if "refine_sentences" in prompt:
        take("refine_sentences", bool(prompt["refine_sentences"]))

    if "padding" in merge:
        take("graph_merge_padding", str(merge["padding"]))
    if "min_group_size" in merge:
        take("graph_merge_min_group_size", str(merge["min_group_size"]))
    if "threshold" in merge:
        take("graph_merge_threshold", str(merge["threshold"]))

    take("graph_matching_model", matching.get("model"))

    if cfg.get("models") is not None:
        take("models", cfg.get("models"))

    take("images", seg.get("img_dir_gt"))

    if cfg.get("num_gpus") is None:
        ng = _gpus_from_block(clean) or _gpus_from_block(attr) or _gpus_from_block(prompt_llm)
        take("num_gpus", ng)

    j_ng = _gpus_from_block(llm_judge)
    take("judge_num_gpus", j_ng)

    return out


def _coalesce(cli: Any, cfg: Any, default: Any) -> Any:
    """Prefer explicit CLI when *provided* (not None); else config; else default."""
    if cli is not None:
        return cli
    if cfg is not None:
        return cfg
    return default


def _resolve_cfg_path(val: Optional[str], fallback: Path) -> str:
    """Resolve a path from JSON relative to the repo root; fall back to an absolute default."""
    if val:
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return str(p)
    return str(fallback.resolve())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Raised when a pipeline step fails."""
    pass


def parse_steps(steps_str: str) -> List[int]:
    """Parse step ranges like '1-7,8,15-19' into a list of ints."""
    steps = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-"))
            steps.extend(range(start, end + 1))
        else:
            steps.append(int(part))
    return sorted(set(steps))


def resolve_conda_init(conda_init: Optional[str]) -> str:
    """
    Return the shell snippet that initialises conda.
    
    Priority:
      1. Explicit --conda-init flag
      2. .temporary_conda/conda_init.sh written by setup.sh (v1)
      3. .tmp/distros/<latest>/conda_init.sh written by setup_v2.sh
      4. Generic fallback (source ~/.bashrc + conda shell hook)
    """
    if conda_init:
        return conda_init

    # Try the setup.sh v1 helper
    helper_v1 = REPO_ROOT / ".temporary_conda" / "conda_init.sh"
    if helper_v1.exists():
        print(f"  [conda] Using setup.sh (v1) init: {helper_v1}")
        return f"source {helper_v1}"

    # Try setup_v2.sh distros — pick the most recent one
    distros_dir = REPO_ROOT / ".tmp" / "distros"
    if distros_dir.is_dir():
        # Sort by name (timestamp-based naming means lexicographic = chronological)
        candidates = sorted(distros_dir.iterdir(), reverse=True)
        for candidate in candidates:
            helper_v2 = candidate / "conda_init.sh"
            if helper_v2.exists():
                print(f"  [conda] Using setup_v2.sh init: {helper_v2}")
                return f"source {helper_v2}"

    # Generic fallback
    print("  [conda] WARNING: No conda_init.sh found. Using generic fallback.")
    print("           If envs are not found, pass --conda-init explicitly, e.g.:")
    print(f"           --conda-init 'source {distros_dir}/<your_run>/conda_init.sh'")
    return (
        'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then '
        '  source "$HOME/miniconda3/etc/profile.d/conda.sh"; '
        'elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then '
        '  source "$HOME/anaconda3/etc/profile.d/conda.sh"; '
        "else "
        '  eval "$(conda shell.bash hook 2>/dev/null)"; '
        "fi"
    )


def run_step(
    step_id: int,
    step_label: str,
    cmd: List[str],
    conda_env: str,
    conda_init_snippet: str,
    dry_run: bool = False,
    log_dir: Optional[Path] = None,
) -> None:
    """
    Run a single pipeline step inside the correct conda environment.

    The command is wrapped in a bash -c invocation that:
      1. Sources conda
      2. Activates the right env
      3. cd's to repo root (so relative script paths work)
      4. Runs the actual command

    Output is always streamed to the console. If log_dir is set, it is
    additionally written to a log file. On failure the last 50 lines of
    output are printed for immediate debugging.
    """
    cmd_str = " ".join(str(c) for c in cmd)

    shell_script = (
        f"set -e; "
        f"{conda_init_snippet}; "
        f"conda activate {conda_env}; "
        f'cd "{REPO_ROOT}"; '
        f"{cmd_str}"
    )

    header = f"[Step {step_id}] {step_label}"
    separator = "=" * 70

    print(f"\n{separator}")
    print(f"{header}")
    print(f"  env:  {conda_env}")
    print(f"  cmd:  {cmd_str}")
    print(separator, flush=True)

    if dry_run:
        print("  (dry run — skipped)")
        return

    start = time.time()

    # Determine log path
    log_path = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = step_label.replace(" ", "_").replace("/", "_")
        log_path = log_dir / f"step_{step_id:02d}_{safe_label}.log"

    # Run the command, capturing output so we can both display and save it
    proc = subprocess.run(
        ["bash", "-c", shell_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = proc.stdout or ""

    # Write log file if requested
    if log_path:
        with open(log_path, "w") as f:
            f.write(output)

    if proc.returncode != 0:
        elapsed = time.time() - start

        # Print the tail of output so the user can see what went wrong immediately
        lines = output.strip().splitlines()
        tail_n = 50
        if lines:
            print(f"\n  --- Last {min(tail_n, len(lines))} lines of output ---")
            for line in lines[-tail_n:]:
                print(f"  | {line}")
            print(f"  --- end of output ---")
        else:
            print("\n  (no output captured)")

        msg = f"{header} FAILED after {elapsed:.1f}s (exit code {proc.returncode})"
        if log_path:
            msg += f"\n  Full log: {log_path}"
        raise PipelineError(msg)

    elapsed = time.time() - start
    print(f"  ✓ completed in {elapsed:.1f}s")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Step builders — each returns (label, cmd_list, conda_env)
# ---------------------------------------------------------------------------


def step_1_segmentation_gt(args, run_dir: Path) -> Tuple[str, List[str], str]:
    """Panoptic segmentation of ground-truth images."""
    out_dir = ensure_dir(run_dir / DIR_NAMES[1])
    return (
        "Segmentation (GT)",
        [
            "python", "scripts/pipeline/generate_segmasks_kmax.py",
            "--img-dir", str(args.images),
            "--out-dir", str(out_dir),
            "--psg-meta", str(args.psg_meta),
            "--kmax-path", str(args.kmax_path),
            "--kmax-config", str(args.kmax_config),
            "--kmax-weights", str(args.kmax_weights),
        ],
        args.env_by_step[1],
    )


def step_2_scene_graph_gt(args, run_dir: Path) -> Tuple[str, List[str], str]:
    """Scene graph generation on GT segmentation."""
    anno_path = run_dir / DIR_NAMES[1] / "anno.json"
    out_dir = ensure_dir(run_dir / DIR_NAMES[2])
    return (
        "Scene Graph Generation (GT)",
        [
            "python", "scripts/pipeline/generate_sg.py",
            "--img-dir", str(args.images),
            "--out-dir", str(out_dir),
            "--psg-meta", str(args.psg_meta),
            "--skip-segmentation",
            "--anno-path", str(anno_path),
            "--model-dir", str(args.psg_model_dir),
        ],
        args.env_by_step[2],
    )


def step_3_clean_refine_gt(args, run_dir: Path) -> Tuple[str, List[str], str]:
    """VLM-based relation validation on GT scene graphs."""
    sg_pkl = run_dir / DIR_NAMES[2] / "scene-graph.pkl"
    out_dir = ensure_dir(run_dir / DIR_NAMES[3])
    return (
        "Clean & Refine Relations (GT)",
        [
            "python", "scripts/pipeline/clean_and_refine_relations.py",
            "--input", str(sg_pkl),
            "--images", str(args.images),
            "--output", str(out_dir),
            "--psg-meta", str(args.psg_meta),
            "--model", args.llm_clean_refine,
            "--num-gpus", str(args.num_gpus),
        ],
        args.env_by_step[3],
    )


def step_4_graph_merge_gt(args, run_dir: Path) -> Tuple[str, List[str], str]:
    """Merge overlapping segments of same category."""
    anno_path = run_dir / DIR_NAMES[1] / "anno.json"
    sg_pkl = run_dir / DIR_NAMES[2] / "scene-graph.pkl"
    clean_dir = run_dir / DIR_NAMES[3]
    out_dir = ensure_dir(run_dir / DIR_NAMES[4])
    cmd = [
        "python", "scripts/pipeline/run_graph_merge.py",
        "--anno-json", str(anno_path),
        "--scene-graph-pkl", str(sg_pkl),
        "--out-dir", str(out_dir),
        "--padding", str(args.graph_merge_padding),
        "--min-group-size", str(args.graph_merge_min_group_size),
        "--clean-relations-dir", str(clean_dir),
    ]
    if getattr(args, "graph_merge_threshold", None) is not None:
        cmd.extend(["--threshold", str(args.graph_merge_threshold)])
    return (
        "Graph Merging (GT)",
        cmd,
        args.env_by_step[4],
    )


def step_5_attributes_gt(args, run_dir: Path) -> Tuple[str, List[str], str]:
    """VLM-based attribute extraction on GT objects."""
    anno_json = run_dir / DIR_NAMES[1] / "anno.json"
    seg_dir = run_dir / DIR_NAMES[1]
    sg_dir = run_dir / DIR_NAMES[4]
    out_dir = ensure_dir(run_dir / DIR_NAMES[5])
    return (
        "Attribute Generation (GT)",
        [
            "python", "scripts/pipeline/generate_attributes.py",
            "--img-dir", str(args.images),
            "--output-dir", str(out_dir),
            "--anno-json", str(anno_json),
            "--seg-dir", str(seg_dir),
            "--mapping-json", str(args.category_mapping),
            "--scene-graphs-dir", str(sg_dir),
            "--model", args.llm_attributes,
            "--num-gpus", str(args.num_gpus),
        ],
        args.env_by_step[5],
    )


def step_6_prompt_generation(args, run_dir: Path) -> Tuple[str, List[str], str]:
    """Generate text prompts from GT scene graphs + attributes."""
    ensure_dir(run_dir / DIR_NAMES[6])
    cmd = [
        "python", "scripts/pipeline/generate_prompts.py",
        "--run_dir", str(run_dir),
    ]
    if args.refine_objects:
        cmd.extend([
            "--refine-objects",
            "--img-dir", str(args.images),
            "--seg-dir", str(run_dir / DIR_NAMES[1]),
            "--model", args.llm_prompt,
            "--num-gpus", str(args.num_gpus),
        ])
    if args.refine_sentences:
        cmd.append("--refine-sentences")
        if not args.refine_objects:
            cmd.extend([
                "--model", args.llm_prompt,
                "--num-gpus", str(args.num_gpus),
            ])
    return ("Prompt Generation", cmd, args.env_by_step[6])


def step_7_question_generation(args, run_dir: Path) -> Tuple[str, List[str], str]:
    """Generate VQA questions from GT scene graphs."""
    sg_with_attrs = run_dir / "sg_with_attributes"
    out_dir = ensure_dir(run_dir / DIR_NAMES[7])
    output_file = out_dir / "generated_questions.json"
    return (
        "Question Generation",
        [
            "python", "scripts/evaluation/question_generation/generate_questions.py",
            "--validated_relations_dir", str(sg_with_attrs),
            "--metadata_file", str(args.vqa_metadata),
            "--synonyms_json", str(args.vqa_synonyms),
            "--template_dir", str(args.vqa_templates),
            "--output_questions_file", str(output_file),
        ],
        args.env_by_step[7],
    )


# --- Per-model prediction steps ---

def step_8_image_generation(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Generate images from prompts using a VLM."""
    prompt_path = run_dir / DIR_NAMES[6] / "prompts.json"
    out_dir = ensure_dir(run_dir / DIR_NAMES[8] / model)
    return (
        f"Image Generation ({model})",
        [
            "python", "scripts/pipeline/generate_images.py",
            "--prompts-json", str(prompt_path),
            "--output-dir", str(out_dir),
            "--models", model,
        ],
        args.env_by_step[8],
    )


def step_9_segmentation_pt(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Panoptic segmentation of model-generated images."""
    img_dir = run_dir / DIR_NAMES[8] / model
    out_dir = ensure_dir(run_dir / DIR_NAMES[9] / model)
    return (
        f"Segmentation (predicted, {model})",
        [
            "python", "scripts/pipeline/generate_segmasks_kmax.py",
            "--img-dir", str(img_dir),
            "--out-dir", str(out_dir),
            "--psg-meta", str(args.psg_meta),
            "--kmax-path", str(args.kmax_path),
            "--kmax-config", str(args.kmax_config),
            "--kmax-weights", str(args.kmax_weights),
        ],
        args.env_by_step[9],
    )


def step_10_scene_graph_pt(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Scene graph generation on predicted segmentation."""
    img_dir = run_dir / DIR_NAMES[8] / model
    anno_path = run_dir / DIR_NAMES[9] / model / "anno.json"
    out_dir = ensure_dir(run_dir / DIR_NAMES[10] / model)
    return (
        f"Scene Graph Generation (predicted, {model})",
        [
            "python", "scripts/pipeline/generate_sg.py",
            "--img-dir", str(img_dir),
            "--out-dir", str(out_dir),
            "--psg-meta", str(args.psg_meta),
            "--skip-segmentation",
            "--anno-path", str(anno_path),
            "--model-dir", str(args.psg_model_dir),
        ],
        args.env_by_step[10],
    )


def step_11_clean_refine_pt(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """VLM relation validation on predicted scene graphs (with lenient spatial prompts)."""
    img_dir = run_dir / DIR_NAMES[8] / model
    sg_pkl = run_dir / DIR_NAMES[10] / model / "scene-graph.pkl"
    out_dir = ensure_dir(run_dir / DIR_NAMES[11] / model)
    return (
        f"Clean & Refine Relations (predicted, {model})",
        [
            "python", "scripts/pipeline/clean_and_refine_relations.py",
            "--input", str(sg_pkl),
            "--images", str(img_dir),
            "--output", str(out_dir),
            "--psg-meta", str(args.psg_meta),
            "--model", args.llm_clean_refine,
            "--num-gpus", str(args.num_gpus),
            "--use-flexible-spatial-prompt",
        ],
        args.env_by_step[11],
    )


def step_12_graph_merge_pt(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Merge predicted segments."""
    anno_path = run_dir / DIR_NAMES[9] / model / "anno.json"
    sg_pkl = run_dir / DIR_NAMES[10] / model / "scene-graph.pkl"
    clean_dir = run_dir / DIR_NAMES[11] / model
    out_dir = ensure_dir(run_dir / DIR_NAMES[12] / model)
    cmd = [
        "python", "scripts/pipeline/run_graph_merge.py",
        "--anno-json", str(anno_path),
        "--scene-graph-pkl", str(sg_pkl),
        "--out-dir", str(out_dir),
        "--padding", str(args.graph_merge_padding),
        "--min-group-size", str(args.graph_merge_min_group_size),
        "--clean-relations-dir", str(clean_dir),
    ]
    if getattr(args, "graph_merge_threshold", None) is not None:
        cmd.extend(["--threshold", str(args.graph_merge_threshold)])
    return (
        f"Graph Merging (predicted, {model})",
        cmd,
        args.env_by_step[12],
    )


def step_13_attributes_pt(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """VLM attribute extraction on predicted objects."""
    img_dir = run_dir / DIR_NAMES[8] / model
    anno_json = run_dir / DIR_NAMES[9] / model / "anno.json"
    seg_dir = run_dir / DIR_NAMES[9] / model
    sg_dir = run_dir / DIR_NAMES[12] / model
    out_dir = ensure_dir(run_dir / DIR_NAMES[13] / model)
    return (
        f"Attribute Generation (predicted, {model})",
        [
            "python", "scripts/pipeline/generate_attributes.py",
            "--img-dir", str(img_dir),
            "--output-dir", str(out_dir),
            "--anno-json", str(anno_json),
            "--seg-dir", str(seg_dir),
            "--mapping-json", str(args.category_mapping),
            "--scene-graphs-dir", str(sg_dir),
            "--model", args.llm_attributes,
            "--num-gpus", str(args.num_gpus),
            "--is-prediction-step",
        ],
        args.env_by_step[13],
    )


def step_14_graph_matching(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Hungarian matching between GT and predicted scene graphs."""
    gt_sg_dir = run_dir / DIR_NAMES[4]
    gt_attr_file = run_dir / DIR_NAMES[5] / "attributes.json"
    pred_sg_dir = run_dir / DIR_NAMES[12] / model
    pred_attr_file = run_dir / DIR_NAMES[13] / model / "attributes.json"
    out_dir = ensure_dir(run_dir / DIR_NAMES[14] / model)
    cmd = [
        "python", "scripts/pipeline/run_graph_matching.py",
        "--gt-sg-dir", str(gt_sg_dir),
        "--gt-attr-file", str(gt_attr_file),
        "--pred-sg-dir", str(pred_sg_dir),
        "--pred-attr-file", str(pred_attr_file),
        "--out-dir", str(out_dir),
    ]
    if getattr(args, "graph_matching_model", None):
        cmd.extend(["--model", str(args.graph_matching_model)])
    return (
        f"Graph Matching ({model})",
        cmd,
        args.env_by_step[14],
    )


def step_15_sg_eval(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Image Generation Evaluation — look up predicted answers from matched graphs."""
    base_dir = run_dir / DIR_NAMES[14] / model
    questions_file = run_dir / DIR_NAMES[7] / "generated_questions.json"
    matched_graphs = base_dir / "scene_graphs_matched.json"
    matching_graphs = base_dir / "scene_graphs_for_matching.json"
    pred_sg_dir = base_dir / "injected_pred"
    output_jsonl = base_dir / f"ige_answers_{model}.jsonl"
    return (
        f"Scene Graph Eval / IGE ({model})",
        [
            "python", "scripts/evaluation/image_generation_eval.py",
            "--questions", str(questions_file),
            "--matched-graphs", str(matched_graphs),
            "--matching-graphs", str(matching_graphs),
            "--pred-sg-dir", str(pred_sg_dir),
            "--output", str(output_jsonl),
        ],
        args.env_by_step[15],
    )


def step_16_llm_judge_sg(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """LLM judge scoring of IGE answers."""
    base_dir = run_dir / DIR_NAMES[14] / model
    pred_jsonl = base_dir / f"ige_answers_{model}.jsonl"
    out_scored = base_dir / f"ige_scored_{model}.jsonl"
    out_metrics = base_dir / f"ige_metrics_{model}.json"
    return (
        f"LLM Judge - SG ({model})",
        [
            "python", "scripts/evaluation/llm_judge.py",
            "--pred_jsonl", str(pred_jsonl),
            "--out_scored_jsonl", str(out_scored),
            "--out_metrics", str(out_metrics),
            "--model", args.llm_judge,
            "--num_gpus", str(args.judge_num_gpus),
            "--mode", "local",
        ],
        args.env_by_step[16],
    )


def step_17_vqa_generation(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Ask VLM questions about generated images."""
    questions_file = run_dir / DIR_NAMES[7] / "generated_questions.json"
    img_dir = run_dir / DIR_NAMES[8] / model
    out_dir = ensure_dir(run_dir / "vqa_outputs" / model / "chunk_0")
    return (
        f"VQA Generation ({model})",
        [
            "python", "scripts/evaluation/run_vqa_benchmark.py",
            "--generated_questions", str(questions_file),
            "--image_dir", str(img_dir),
            "--output_dir", str(out_dir),
            "--models", model,
            "--batch_size", "1",
        ],
        args.env_by_step[17],
    )


def step_18_vqa_merge(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """Merge chunked VQA results."""
    out_base = run_dir / "vqa_outputs" / model
    merged_output = out_base / f"merged_{model}.jsonl"
    gt_pattern = str(out_base / "chunk_*/gt_index.jsonl")
    pred_pattern = str(out_base / f"chunk_*/{model}_answers.gpu*.jsonl")
    return (
        f"VQA Merge ({model})",
        [
            "python", "scripts/evaluation/merge_results.py",
            "--gt_index", gt_pattern,
            "--pred_files", pred_pattern,
            "--output", str(merged_output),
        ],
        args.env_by_step[18],
    )


def step_19_llm_judge_vqa(args, run_dir: Path, model: str) -> Tuple[str, List[str], str]:
    """LLM judge scoring of VQA answers."""
    base_dir = run_dir / "vqa_outputs" / model
    pred_jsonl = base_dir / f"merged_{model}.jsonl"
    out_scored = base_dir / f"scored_{model}.jsonl"
    out_metrics = base_dir / f"metrics_{model}.json"
    return (
        f"LLM Judge - VQA ({model})",
        [
            "python", "scripts/evaluation/llm_judge.py",
            "--pred_jsonl", str(pred_jsonl),
            "--out_scored_jsonl", str(out_scored),
            "--out_metrics", str(out_metrics),
            "--model", args.llm_judge,
            "--num_gpus", str(args.judge_num_gpus),
            "--mode", "local",
        ],
        args.env_by_step[19],
    )



# ---------------------------------------------------------------------------
# GT dataset import (when --gt-dataset is provided)
# ---------------------------------------------------------------------------

def import_gt_dataset(gt_dataset_path: Path, run_dir: Path) -> None:
    """
    When the user provides a pre-computed GT dataset, symlink or copy the
    expected directories into run_dir so that prediction steps can find them.

    Expected structure inside gt_dataset_path:
      1_segmentation_gt/
      2_scene_graphs_gt/        (optional — not needed by prediction steps)
      3_clean_and_refine_gt/    (optional)
      4_graph_merge_gt/
      5_attributes_gt/
      6_prompt_generation/      (must contain prompts.json)
      vqa_questions/            (must contain generated_questions.json)
      sg_with_attributes/       (used by question generation)
    """
    print(f"\n{'=' * 70}")
    print(f"Importing pre-computed GT dataset from: {gt_dataset_path}")
    print(f"{'=' * 70}")

    required = [
        "1_segmentation_gt",
        "4_graph_merge_gt",
        "5_attributes_gt",
        "6_prompt_generation",
        "vqa_questions",
    ]
    optional = [
        "2_scene_graphs_gt",
        "3_clean_and_refine_gt",
        "sg_with_attributes",
    ]

    missing = [d for d in required if not (gt_dataset_path / d).exists()]
    if missing:
        raise PipelineError(
            f"GT dataset is missing required directories: {missing}\n"
            f"Expected them inside: {gt_dataset_path}"
        )

    for dirname in required + optional:
        src = gt_dataset_path / dirname
        dst = run_dir / dirname
        if src.exists() and not dst.exists():
            # Symlink to avoid copying large data
            dst.symlink_to(src.resolve())
            print(f"  Linked: {dirname}")
        elif dst.exists():
            print(f"  Already exists: {dirname}")
        else:
            print(f"  Not found (optional): {dirname}")

    print("  ✓ GT dataset imported successfully")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def build_execution_plan(
    args,
    run_dir: Path,
    allowed_steps: List[int],
    models: List[str],
) -> List[Tuple[int, str, List[str], str]]:
    """
    Build the ordered list of (step_id, label, cmd, conda_env) to execute.
    """
    plan = []

    def add(step_id, builder_result):
        if step_id in allowed_steps:
            label, cmd, env = builder_result
            plan.append((step_id, label, cmd, env))

    # --- GT steps (1-7) ---
    if not args.gt_dataset:
        add(1, step_1_segmentation_gt(args, run_dir))
        add(2, step_2_scene_graph_gt(args, run_dir))
        add(3, step_3_clean_refine_gt(args, run_dir))
        add(4, step_4_graph_merge_gt(args, run_dir))
        add(5, step_5_attributes_gt(args, run_dir))
        add(6, step_6_prompt_generation(args, run_dir))
        add(7, step_7_question_generation(args, run_dir))
    else:
        if any(s in allowed_steps for s in range(1, 8)):
            print("\n  NOTE: Steps 1-7 skipped because --gt-dataset was provided.")

    # --- Per-model prediction steps (8-19) ---
    for model in models:
        add(8,  step_8_image_generation(args, run_dir, model))
        add(9,  step_9_segmentation_pt(args, run_dir, model))
        add(10, step_10_scene_graph_pt(args, run_dir, model))
        add(11, step_11_clean_refine_pt(args, run_dir, model))
        add(12, step_12_graph_merge_pt(args, run_dir, model))
        add(13, step_13_attributes_pt(args, run_dir, model))
        add(14, step_14_graph_matching(args, run_dir, model))
        add(15, step_15_sg_eval(args, run_dir, model))
        add(16, step_16_llm_judge_sg(args, run_dir, model))
        add(17, step_17_vqa_generation(args, run_dir, model))
        add(18, step_18_vqa_merge(args, run_dir, model))
        add(19, step_19_llm_judge_vqa(args, run_dir, model))

    return plan


def main():
    parser = argparse.ArgumentParser(
        description="XTC-Bench: Run the full evaluation pipeline sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required (or supply via --config JSON: images / output_dir) ---
    parser.add_argument(
        "--images", type=str, default=argparse.SUPPRESS,
        help="Directory containing input images (e.g. COCO val2017). "
             "May be omitted if set in the config file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=argparse.SUPPRESS,
        help="Output directory for this pipeline run. May be omitted if set in the config file.",
    )

    # --- Optional: GT shortcut ---
    parser.add_argument(
        "--gt-dataset", type=str, default=argparse.SUPPRESS,
        help="Path to pre-computed GT dataset. If provided, steps 1-7 are skipped.",
    )

    # --- Models ---
    parser.add_argument(
        "--models", nargs="+", default=argparse.SUPPRESS,
        help="VLM model(s) to evaluate (e.g. showo2 blip3o januspro mmada omnigen2).",
    )

    # --- Per-step LLM configuration ---
    # Omitted flags are filled from --config (pipeline JSON or flat JSON). Explicit
    # CLI values always win. For pipeline JSON, models/paths come from the same
    # nested blocks as run_full_pipeline.py / pipeline_factory.py.
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (pipeline layout or flat keys). Used for any "
             "option not set on the command line.",
    )
    parser.add_argument(
        "--llm-clean-refine", type=str, default=argparse.SUPPRESS,
        help="VLM model for relation validation (steps 3, 11). "
             "E.g. Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--llm-attributes", type=str, default=argparse.SUPPRESS,
        help="VLM model for attribute generation (steps 5, 13). "
             "E.g. Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--llm-prompt", type=str, default=argparse.SUPPRESS,
        help="VLM model for prompt/object description refinement (step 6). "
             "E.g. Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--llm-judge", type=str, default=argparse.SUPPRESS,
        help="LLM model for scoring answers (steps 16, 19). "
             "E.g. Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=argparse.SUPPRESS,
        help="Number of GPUs for VLM steps (clean/refine, attributes, prompt).",
    )

    # --- Step selection ---
    parser.add_argument(
        "--steps", type=str, default=argparse.SUPPRESS,
        help="Steps to run (e.g. '1-7', '8-19', '1-5,8,14-19'). Default: 1-19.",
    )

    # --- Execution options ---
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--conda-init", type=str, default=argparse.SUPPRESS,
        help="Shell command to initialise conda (e.g. 'source ~/miniconda3/bin/activate'). "
             "Auto-detected if not set.",
    )
    parser.add_argument(
        "--refine-sentences", action="store_true",
        help="Enable LLM-based sentence refinement in step 6.",
    )
    parser.add_argument(
        "--no-refine-objects", action="store_true",
        help="Disable LLM-based per-object description refinement in step 6. "
             "Useful if the GenerateAnyScene env has compatibility issues.",
    )
    parser.add_argument(
        "--save-logs", action="store_true",
        help="Save stdout/stderr of each step to <output-dir>/logs/.",
    )

    args = parser.parse_args()

    DEFAULT_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
    DEFAULT_JUDGE = "Qwen/Qwen2.5-7B-Instruct"
    DEFAULT_GPUS = 1

    file_cfg: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            file_cfg = json.load(f)
        print(f"  Loaded config from: {config_path}")

    eff = extract_settings_from_json(file_cfg)

    # --- Merge: explicit CLI > config JSON > built-in defaults ---
    args.images = _coalesce(getattr(args, "images", None), eff.get("images"), None)
    args.output_dir = _coalesce(getattr(args, "output_dir", None), eff.get("output_dir"), None)
    if not args.images or not args.output_dir:
        print(
            "ERROR: --images and --output-dir are required (or set both as top-level "
            "'images' and 'output_dir' / 'run_dir' in the config file)."
        )
        sys.exit(1)

    args.gt_dataset = _coalesce(getattr(args, "gt_dataset", None), eff.get("gt_dataset"), None)
    args.models = _coalesce(getattr(args, "models", None), eff.get("models"), ["showo2"])
    if isinstance(args.models, str):
        args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.steps = _coalesce(getattr(args, "steps", None), eff.get("steps"), "1-19")
    args.conda_init = _coalesce(getattr(args, "conda_init", None), eff.get("conda_init"), None)

    args.llm_clean_refine = _coalesce(
        getattr(args, "llm_clean_refine", None), eff.get("llm_clean_refine"), DEFAULT_VLM
    )
    args.llm_attributes = _coalesce(
        getattr(args, "llm_attributes", None), eff.get("llm_attributes"), DEFAULT_VLM
    )
    args.llm_prompt = _coalesce(
        getattr(args, "llm_prompt", None), eff.get("llm_prompt"), DEFAULT_VLM
    )
    args.llm_judge = _coalesce(
        getattr(args, "llm_judge", None), eff.get("llm_judge"), DEFAULT_JUDGE
    )
    args.num_gpus = int(_coalesce(getattr(args, "num_gpus", None), eff.get("num_gpus"), DEFAULT_GPUS))

    j_gpus = eff.get("judge_num_gpus")
    args.judge_num_gpus = int(j_gpus) if j_gpus is not None else args.num_gpus

    if hasattr(args, "no_refine_objects") and args.no_refine_objects:
        args.refine_objects = False
    elif "refine_objects" in eff:
        args.refine_objects = bool(eff["refine_objects"])
    else:
        args.refine_objects = True

    args.refine_sentences = bool(
        getattr(args, "refine_sentences", False) or eff.get("refine_sentences", False)
    )
    args.dry_run = bool(getattr(args, "dry_run", False) or eff.get("dry_run", False))
    args.save_logs = bool(getattr(args, "save_logs", False) or eff.get("save_logs", False))

    args.psg_meta = _resolve_cfg_path(eff.get("psg_meta_path"), DEFAULT_PSG_META)
    args.kmax_path = _resolve_cfg_path(eff.get("kmax_path"), DEFAULT_KMAX_SUBMODULE)
    args.kmax_config = _resolve_cfg_path(eff.get("kmax_config_path"), DEFAULT_KMAX_CONFIG)
    args.kmax_weights = _resolve_cfg_path(eff.get("kmax_weights_path"), DEFAULT_KMAX_WEIGHTS)
    args.psg_model_dir = _resolve_cfg_path(eff.get("psg_model_dir"), DEFAULT_PSG_MODEL_DIR)
    args.category_mapping = _resolve_cfg_path(eff.get("category_mapping_path"), DEFAULT_CATEGORY_MAPPING)
    args.vqa_metadata = _resolve_cfg_path(eff.get("vqa_metadata_path"), DEFAULT_VQA_METADATA)
    args.vqa_synonyms = _resolve_cfg_path(eff.get("vqa_synonyms_path"), DEFAULT_VQA_SYNONYMS)
    args.vqa_templates = _resolve_cfg_path(eff.get("vqa_templates_dir"), DEFAULT_VQA_TEMPLATES)

    args.graph_merge_padding = eff.get("graph_merge_padding", "10")
    args.graph_merge_min_group_size = eff.get("graph_merge_min_group_size", "3")
    args.graph_merge_threshold = eff.get("graph_merge_threshold")
    args.graph_matching_model = eff.get("graph_matching_model")

    # Per-step conda env: defaults from STEP_ENVS, overridden by pipeline JSON or step_conda_envs
    args.env_by_step = {**STEP_ENVS, **collect_step_conda_envs_from_cfg(file_cfg)}

    # Resolve paths
    args.images = str(Path(args.images).resolve())
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.gt_dataset:
        args.gt_dataset = Path(args.gt_dataset).resolve()

    allowed_steps = parse_steps(args.steps)
    conda_init = resolve_conda_init(args.conda_init)
    log_dir = run_dir / "logs" if args.save_logs else None

    # Print configuration summary
    print("=" * 70)
    print("XTC-Bench Sequential Pipeline")
    print("=" * 70)
    print(f"  Images:          {args.images}")
    print(f"  Output:          {run_dir}")
    print(f"  GT dataset:      {args.gt_dataset or '(will generate)'}")
    print(f"  Models:          {', '.join(args.models)}")
    print(f"  LLM clean/refine:{args.llm_clean_refine}")
    print(f"  LLM attributes:  {args.llm_attributes}")
    print(f"  LLM prompt:      {args.llm_prompt}")
    print(f"  LLM judge:       {args.llm_judge}")
    print(f"  GPUs (VLM):     {args.num_gpus}")
    print(f"  GPUs (judge):   {args.judge_num_gpus}")
    conda_diff = {
        sid: args.env_by_step[sid]
        for sid in range(1, 20)
        if args.env_by_step.get(sid) != STEP_ENVS.get(sid)
    }
    if conda_diff:
        print(f"  Conda envs:      defaults overridden for steps {conda_diff}")
    print(f"  Steps:           {args.steps} -> {allowed_steps}")
    print(f"  Dry run:         {args.dry_run}")
    print(f"  Logs:            {log_dir or '(console only)'}")
    print("=" * 70)

    # Import GT if provided
    if args.gt_dataset:
        import_gt_dataset(args.gt_dataset, run_dir)

    # Save run configuration for reproducibility
    config_record = {
        "images": args.images,
        "output_dir": str(run_dir),
        "gt_dataset": str(args.gt_dataset) if args.gt_dataset else None,
        "models": args.models,
        "llm_clean_refine": args.llm_clean_refine,
        "llm_attributes": args.llm_attributes,
        "llm_prompt": args.llm_prompt,
        "llm_judge": args.llm_judge,
        "num_gpus": args.num_gpus,
        "judge_num_gpus": args.judge_num_gpus,
        "psg_meta": args.psg_meta,
        "kmax_path": args.kmax_path,
        "kmax_config": args.kmax_config,
        "kmax_weights": args.kmax_weights,
        "psg_model_dir": args.psg_model_dir,
        "category_mapping": args.category_mapping,
        "vqa_metadata": args.vqa_metadata,
        "vqa_synonyms": args.vqa_synonyms,
        "vqa_templates": args.vqa_templates,
        "graph_merge_padding": args.graph_merge_padding,
        "graph_merge_min_group_size": args.graph_merge_min_group_size,
        "graph_merge_threshold": args.graph_merge_threshold,
        "graph_matching_model": args.graph_matching_model,
        "conda_env_by_step": {str(k): v for k, v in sorted(args.env_by_step.items())},
        "refine_objects": args.refine_objects,
        "refine_sentences": args.refine_sentences,
        "steps": args.steps,
        "allowed_steps": allowed_steps,
    }
    config_path = run_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=2)

    # Build and execute plan
    plan = build_execution_plan(args, run_dir, allowed_steps, args.models)

    if not plan:
        print("\nNo steps to execute. Check --steps and --models.")
        return

    print(f"\nExecution plan: {len(plan)} step(s)")
    for step_id, label, _, env in plan:
        print(f"  [{step_id:2d}] {label}  (env: {env})")

    total_start = time.time()
    completed = 0
    failed = 0

    for step_id, label, cmd, env in plan:
        try:
            run_step(
                step_id=step_id,
                step_label=label,
                cmd=cmd,
                conda_env=env,
                conda_init_snippet=conda_init,
                dry_run=args.dry_run,
                log_dir=log_dir,
            )
            completed += 1
        except PipelineError as exc:
            print(f"\n  ✗ {exc}")
            failed += 1
            print(f"\n  Pipeline stopped at step {step_id}. Fix the issue and re-run with --steps '{step_id}-19'.")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print(f"Pipeline {'DRY RUN ' if args.dry_run else ''}complete!")
    print(f"  Steps completed: {completed}")
    print(f"  Total time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"  Results in:      {run_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()