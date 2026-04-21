# XTC-Bench

This repository contains the code for **"XTC-Bench: Evaluating Cross-Task Visual Semantic Consistency in Unified Multimodal Models."**

XTC-Bench is a scene-graph-grounded evaluation framework that measures cross-task visual semantic consistency. It automates the entire evaluation pipeline-from ground truth scene graph extraction to VLM image generation and final automated LLM judge scoring.

## Quickstart (Recommended)

This quickstart is the fastest path to a local end-to-end run without SLURM. It runs sequentially on a single machine and evaluates only `januspro`.

1. Initialize submodules:

```bash
git submodule update --init --recursive
```

This fetches required code in `submodules/` (for example `kmax-deeplab`, `panoptic-scene-graph-generation`, and `univlm`) used by the pipeline.

2. Run core setup (creates conda envs and auto-downloads required core weights):

```bash
./setup/setup.sh --install-target temporary --recreate
```

This creates isolated conda environments under `.tmp/distros/...`, initializes dependencies, downloads required core model artifacts, and prints a `conda_init.sh` helper path for config wiring.

3. Install baseline generation stack for a JanusPro-only run:

```bash
SKIP_MODELS="blip3o,showo,showo2,mmada,omnigen2,tar,bagel,emu3" ./setup/setup_baselines.sh --recreate
```

This prepares the baseline environment and downloads JanusPro while skipping other baseline model downloads.

4. Download precomputed GT datasets (COCO and VG):

```bash
./setup/download_xtc_dataset.sh
```

This pulls dataset bundles from `XTC-Bench/xtc-dataset` into `datasets/xtc-dataset/` with:

- `gt-1000-coco/`
- `gt-1000-vg/`

Each bundle already contains the GT run structure expected by `run_sequential.py`:

- `1_segmentation_gt`
- `2_scene_graphs_gt`
- `3_clean_and_refine_gt`
- `4_graph_merge_gt`
- `5_attributes_gt`
- `6_prompt_generation`
- `images`
- `sg_with_attributes`
- `vqa_questions`

5. Run sequential benchmark with GT bundle as the run input:

COCO GT bundle:

```bash
python run_sequential.py --config configs/config_quickstart_januspro.json
```

VG GT bundle:

```bash
python run_sequential.py --config configs/config_quickstart_januspro.json --images datasets/xtc-dataset/gt-1000-vg/images --gt-dataset datasets/xtc-dataset/gt-1000-vg --output-dir outputs/run_1000_vg
```

These commands import the GT folders into each run directory and execute prediction/evaluation steps (`8-19`) for `januspro`.

6. Generate a combined JanusPro evaluation table (COCO + VG):

After step 5, your run directories should look like:

```text
outputs/run_1000_coco/
  final_graphs_pt/januspro/ige_scored_*.jsonl
  vqa_outputs/januspro/scored_*.jsonl

outputs/run_1000_vg/
  final_graphs_pt/januspro/ige_scored_*.jsonl
  vqa_outputs/januspro/scored_*.jsonl
```

```bash
python scripts/evaluation/generate_evaluation_tables.py --run-dir outputs/run_1000_coco --run-dir outputs/run_1000_vg --models januspro --output outputs/tables_coco_vg_januspro.txt
```

This writes a consolidated report to `outputs/tables_coco_vg_januspro.txt`.

Config files for sequential runs:

- `configs/config_quickstart_januspro.json`: prefilled quickstart config (JanusPro + COCO defaults, no SLURM fields)
- `configs/config_sequential_template.json`: non-SLURM template with placeholders for custom setups

## Setup Details

### Core setup (`setup/setup.sh`)

`setup/setup.sh` is the main setup entrypoint. By default it:

- Creates benchmark conda environments from `setup/environments/*.yml`
- Initializes benchmark submodules
- Downloads required core model artifacts via `setup/download_models.sh`
- Prints a generated conda init helper path for config wiring

Recommended default (isolated and reproducible):

```bash
./setup/setup.sh --install-target temporary --recreate
```

Advanced mode (reuse an existing conda installation):

```bash
./setup/setup.sh --install-target existing --existing-conda-root "$(conda info --base)" --recreate
```

Useful options:

- `--skip-weights`: skip automatic core weight download
- `--only env1,env2`: create only selected envs
- `--skip-smoke`: skip import smoke checks
- `--blackwell`: use `scripts/pipeline/generate_any_scene/environment_blackwell.yml`

### Baseline setup (`setup/setup_baselines.sh`) (optional)

Use this for the image-generation baseline stack (BLIP3o, Show-o/Show-o2, MMaDA, JanusPro, TAR, Bagel, OmniGen2).

```bash
./setup/setup_baselines.sh --recreate
```

This script:

- Creates/reuses a dedicated baseline conda env
- Downloads BLIP3o model artifacts
- Downloads Wan2.1 VAE for Show-o2
- Pre-downloads additional HF model repos referenced by the pipeline
- Generates a Show-o2 config override when template is available

To skip selected model downloads (use valid model keys):

```bash
SKIP_MODELS="blip3o,januspro,mmada,omnigen2,tar,bagel,showo" ./setup/setup_baselines.sh --recreate
```

If you use an external UniVLM checkout:

```bash
UNIVLM_PATH=/absolute/path/to/univlm ./setup/setup_baselines.sh --recreate
```

### Manual fallback (only if needed)

If automatic downloads are blocked in your environment, use:

```bash
./setup/download_models.sh --force
```

For GT dataset bundles, you can also choose a subset during download:

```bash
./setup/download_xtc_dataset.sh --subset coco
./setup/download_xtc_dataset.sh --subset vg
```

Expected core outputs:

- `weights/kmax_convnext_large.pth`
- `weights/models/masks-loc-sem/`

## Configuration Essentials

Start from `configs/config_test_1000.json` and replace placeholders.

Key fields:

- `conda_init_script`: set this to the helper printed by `setup/setup.sh` (for example `source <...>/conda_init.sh`)
- `segmentation.img_dir_gt` and other required data/model paths
- `models`: list of image-generation models to run
- Per-step `slurm` blocks (`account`, `partition`, `gpus`, `mem`, `time`, `conda_env`)

For baseline path/config overrides, use:

- `evaluation.shared_model_config.model_paths.*`
- `evaluation.shared_model_config.config_paths.*`

## Running the Benchmark

### SLURM DAG mode

Run the full DAG (default pipeline path):

```bash
python run_benchmark.py \
  --config configs/config_test_1000.json \
  --run-dir outputs/run_1000 \
  --steps "1-19"
```

Run only selected steps:

```bash
python run_benchmark.py --config configs/config_test_1000.json --run-dir outputs/run_1000 --steps "6,7,17-19"
```

### Sequential mode (single machine)

Run step-by-step on one machine:

```bash
python run_sequential.py \
  --config configs/config_test_1000.json \
  --images /path/to/images \
  --output-dir outputs/my_run \
  --models januspro showo2 blip3o \
  --steps 1-19 \
  --llm-clean-refine Qwen/Qwen2.5-VL-7B-Instruct \
  --llm-attributes Qwen/Qwen2.5-VL-7B-Instruct \
  --llm-prompt Qwen/Qwen2.5-VL-7B-Instruct \
  --llm-judge Qwen/Qwen2.5-7B-Instruct \
  --num-gpus 1 \
  --conda-init 'source /path/to/miniconda3/etc/profile.d/conda.sh' \
  --save-logs
```

If reusing precomputed GT, you can run only prediction/eval steps:

```bash
python run_sequential.py \
  --config configs/config_test_1000.json \
  --images /path/to/images \
  --output-dir outputs/my_run \
  --gt-dataset /path/to/precomputed_gt_bundle \
  --models januspro showo2 blip3o \
  --steps 8-19 \
  --save-logs
```

## Evaluation Tables

Use `scripts/evaluation/generate_evaluation_tables.py` to summarize completed runs.

```bash
python scripts/evaluation/generate_evaluation_tables.py --run-dir outputs/run_1000_coco --run-dir outputs/run_1000_vg --models januspro --output outputs/tables_coco_vg_januspro.txt
```

Each `--run-dir` must contain `final_graphs_pt/<model>/ige_scored_*.jsonl` and `vqa_outputs/<model>/scored_*.jsonl` (for quickstart: `<model>` is `januspro`). The output report contains Tables 0-5. If scores are missing, rerun steps `16-19` (or `8-19`).

## SLURM Configuration Reference

Each step section can define a `slurm` dictionary. Keys map directly to `sbatch` arguments:

- `account`: Slurm billing account
- `partition`: cluster partition (for example `gpu-batch`)
- `constraint`: hardware constraint (for example `GPU_SKU:H100`)
- `gpus`: number of GPUs
- `mem`: memory per node
- `time`: wall time limit (for example `120:00:00`)
- `conda_env`: environment name to activate for the step

Example:

```json
"segmentation": {
  "slurm": {
    "account": "your_account",
    "partition": "gpu-batch",
    "constraint": "GPU_SKU:A100",
    "gpus": 1,
    "mem": "32G",
    "time": "32:00:00",
    "conda_env": "kmax_env"
  }
}
```

## Notes on Gated Models and Storage

Some baseline wrappers may require gated Hugging Face models (for example dependencies used by TAR). See `submodules/univlm/README.md` for the current UniVLM-specific access/setup instructions.

Pre-downloading all baseline models can require hundreds of GB. Use `SKIP_MODELS` if you only need a subset.
