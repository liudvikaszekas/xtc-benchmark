#!/usr/bin/env bash

set -eo pipefail

usage() {
    cat <<'EOF'
Usage: setup/setup_baselines.sh [--recreate]

Sets up the baseline generation stack used by benchmark image generation models
(BLIP3o, Show-o/Show-o2, MMaDA, JanusPro, TAR, Bagel, OmniGen2).

Environment overrides:
  BASE_STORAGE           Root for env/cache/models (default: <repo>/.tmp/baselines)
  CONDA_ENV_PATH         Baseline conda env path (default: $BASE_STORAGE/envs/univlm)
  ENV_YAML               Conda YAML to create env (default: setup/environments/univlm_env.yml)
    UNIVLM_PATH            UniVLM checkout path (default: submodules/univlm)
  PIPELINE_DIR           Pipeline scripts path (default: scripts/pipeline)
  MODELS_DIR             Models storage dir (default: $BASE_STORAGE/models)
  HF_HOME                HF cache (default: $BASE_STORAGE/hf_cache)
  TRITON_CACHE_DIR       Triton cache (default: $BASE_STORAGE/triton_cache)
  SKIP_MODELS            Comma-separated list of models to skip downloading (e.g. blip3o,januspro,mmada)
  SHOWO2_CONFIG_TEMPLATE Optional source showo2 config template path
  SHOWO2_CONFIG_OUTPUT   Output showo2 config override path (default: $BASE_STORAGE/configs/showo2_config.yaml)

Notes:
    - UniVLM is required for baseline generation runtime imports.
    - This script validates UNIVLM_PATH and optionally initializes its submodules.
    - If SHOWO2_CONFIG_TEMPLATE is provided (or auto-detected from UNIVLM_PATH),
        a patched config is generated that points at the downloaded Wan2.1 VAE.
EOF
}

RECREATE=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == "--recreate" ]]; then
    RECREATE=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_UNIVLM_PATH="${REPO_ROOT}/submodules/univlm"

BASE_STORAGE="${BASE_STORAGE:-${REPO_ROOT}/.tmp/baselines}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${BASE_STORAGE}/envs/univlm}"
ENV_YAML="${ENV_YAML:-${REPO_ROOT}/setup/environments/univlm_env.yml}"
UNIVLM_PATH="${UNIVLM_PATH:-${DEFAULT_UNIVLM_PATH}}"
PIPELINE_DIR="${PIPELINE_DIR:-${REPO_ROOT}/scripts/pipeline}"
MODELS_DIR="${MODELS_DIR:-${BASE_STORAGE}/models}"
SHOWO2_VAE_PATH="${MODELS_DIR}/Wan2.1_VAE.pth"
SHOWO2_CONFIG_OUTPUT="${SHOWO2_CONFIG_OUTPUT:-${BASE_STORAGE}/configs/showo2_config.yaml}"

export HF_HOME="${HF_HOME:-${BASE_STORAGE}/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BASE_STORAGE}/triton_cache}"

mkdir -p "$(dirname "${CONDA_ENV_PATH}")" "${MODELS_DIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "$(dirname "${SHOWO2_CONFIG_OUTPUT}")"

if [[ ! -f "${ENV_YAML}" ]]; then
    echo "Missing environment YAML: ${ENV_YAML}" >&2
    exit 1
fi

if [[ ! -f "${UNIVLM_PATH}/evaluation/roundtrip_factory.py" ]]; then
    echo "Missing UniVLM checkout or invalid path: ${UNIVLM_PATH}" >&2
    echo "Expected file: ${UNIVLM_PATH}/evaluation/roundtrip_factory.py" >&2
    echo "Set UNIVLM_PATH to a valid checkout, e.g.:" >&2
    echo "  UNIVLM_PATH=/abs/path/to/univlm ./setup/setup_baselines.sh --recreate" >&2
    exit 1
fi

if [[ -d "${UNIVLM_PATH}/.git" ]]; then
    echo "=== Initializing UniVLM submodules ==="
    git -C "${UNIVLM_PATH}" submodule update --init --recursive || true
fi

if [[ -z "${SHOWO2_CONFIG_TEMPLATE:-}" && -f "${UNIVLM_PATH}/configs/showo2_config.yaml" ]]; then
    SHOWO2_CONFIG_TEMPLATE="${UNIVLM_PATH}/configs/showo2_config.yaml"
fi

echo "=== Resolving conda ==="
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh under ~/miniconda3 or ~/anaconda3" >&2
    exit 1
fi

echo "=== Creating baseline conda env ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if [[ "${RECREATE}" -eq 1 ]]; then
    echo "Recreate requested: removing ${CONDA_ENV_PATH}"
    conda env remove -p "${CONDA_ENV_PATH}" -y || true
fi

if [[ ! -d "${CONDA_ENV_PATH}" ]]; then
    conda env create -p "${CONDA_ENV_PATH}" -f "${ENV_YAML}"
else
    echo "Reusing existing env: ${CONDA_ENV_PATH}"
fi

conda activate "${CONDA_ENV_PATH}"
conda install -y -c nvidia cuda-toolkit || true

export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

if [[ "${INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
    echo "=== Optional: installing flash-attn (best effort) ==="
    if ! python -m pip install "flash-attn==2.8.3"; then
        echo "Warning: flash-attn install failed. Continuing without it." >&2
        echo "Set INSTALL_FLASH_ATTN=0 to skip this step explicitly." >&2
    fi
fi

python - <<'PY'
import cv2
import tqdm
print(f"Baseline Python deps OK: cv2={cv2.__version__}, tqdm={tqdm.__version__}")
PY

echo "=== Downloading baseline model artifacts ==="
export SKIP_MODELS="${SKIP_MODELS:-}"

if [[ ",${SKIP_MODELS}," == *",blip3o,"* ]]; then
    echo "Skipping BLIP3o-Model-8B download as requested by SKIP_MODELS."
elif [[ ! -d "${MODELS_DIR}/BLIP3o-Model-8B" ]]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BLIP3o/BLIP3o-Model-8B', repo_type='model', local_dir='${MODELS_DIR}/BLIP3o-Model-8B')"
else
    echo "BLIP3o-Model-8B already exists."
fi

MODEL_INDEX="${MODELS_DIR}/BLIP3o-Model-8B/diffusion-decoder/model_index.json"
if [[ -f "${MODEL_INDEX}" ]]; then
    python - "${MODEL_INDEX}" <<'PY'
import json
import sys
import easydict

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

changed = False
if data.get("multimodal_encoder") != [None, None]:
    data["multimodal_encoder"] = [None, None]
    changed = True
if data.get("tokenizer") != [None, None]:
    data["tokenizer"] = [None, None]
    changed = True

if changed:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("Patched BLIP3o diffusion-decoder/model_index.json")
else:
    print("BLIP3o diffusion-decoder/model_index.json already patched")
PY
fi

if [[ ! -f "${SHOWO2_VAE_PATH}" ]]; then
    wget -O "${SHOWO2_VAE_PATH}" "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth"
else
    echo "Wan2.1_VAE.pth already exists."
fi

echo "=== Pre-downloading baseline HF repos referenced by benchmark pipeline ==="
python - <<PY
import sys
import os
from huggingface_hub import snapshot_download

sys.path.insert(0, "${PIPELINE_DIR}")
from generate_images_all_models import MODEL_CONFIGS

skip_models = [m.strip() for m in os.environ.get("SKIP_MODELS", "").split(",") if m.strip()]
print("Inspecting MODEL_CONFIGS and pre-downloading repositories...")
for key, config in MODEL_CONFIGS.items():
    if key in skip_models:
        print(f"Skipping {key} because it is in SKIP_MODELS")
        continue
    if key == "blip3o": # blip3o is already handled in bash section above
        continue
    repo_id = config.get("default_path", "")
    if repo_id and not repo_id.startswith("~") and not repo_id.startswith("/"):
        try:
            print(f"Downloading {config.get('name', key)} from {repo_id}")
            snapshot_download(repo_id=repo_id, repo_type="model", max_workers=8)
        except Exception as e:
            print(f"Failed to download {repo_id}: {e}")
PY

if [[ -n "${SHOWO2_CONFIG_TEMPLATE:-}" ]]; then
    if [[ ! -f "${SHOWO2_CONFIG_TEMPLATE}" ]]; then
        echo "SHOWO2_CONFIG_TEMPLATE set but file not found: ${SHOWO2_CONFIG_TEMPLATE}" >&2
        exit 1
    fi
    python - <<PY
from pathlib import Path
import yaml

src = Path("${SHOWO2_CONFIG_TEMPLATE}")
dst = Path("${SHOWO2_CONFIG_OUTPUT}")
vae = "${SHOWO2_VAE_PATH}"

cfg = yaml.safe_load(src.read_text())
cfg.setdefault("model", {}).setdefault("vae_model", {})["pretrained_model_path"] = vae
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"Wrote showo2 config override: {dst}")
PY
else
    echo "SHOWO2_CONFIG_TEMPLATE not set: skipping showo2 config override generation."
fi

echo "=== Baseline setup completed ==="
echo "UniVLM path: ${UNIVLM_PATH}"
echo "Conda env path: ${CONDA_ENV_PATH}"
echo "Model storage: ${MODELS_DIR}"
echo "HF cache: ${HF_HOME}"
echo "Show-o2 VAE: ${SHOWO2_VAE_PATH}"
if [[ -f "${SHOWO2_CONFIG_OUTPUT}" ]]; then
    echo "Show-o2 config override: ${SHOWO2_CONFIG_OUTPUT}"
fi
echo
echo "Suggested benchmark config wiring:"
echo "- conda_init_script: source ${CONDA_ENV_PATH}/bin/activate"
echo "- shared_model_config.model_paths.blip3o: ${MODELS_DIR}/BLIP3o-Model-8B"
if [[ -f "${SHOWO2_CONFIG_OUTPUT}" ]]; then
    echo "- shared_model_config.config_paths.showo2: ${SHOWO2_CONFIG_OUTPUT}"
fi
