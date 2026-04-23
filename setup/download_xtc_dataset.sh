#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_REPO="XTC-Bench/xtc-dataset"
DEST_DIR="${REPO_ROOT}/datasets/xtc-dataset"
SUBSET="all"
FORCE=0

usage() {
    cat <<'EOF'
Usage: setup/download_xtc_dataset.sh [options]

Download precomputed GT bundles from Hugging Face dataset:
  https://huggingface.co/datasets/XTC-Bench/xtc-dataset

By default downloads both folders:
  - gt-1000-coco
  - gt-1000-vg

Options:
  --dest DIR         Local destination directory
                     (default: datasets/xtc-dataset)
  --subset NAME      Which GT bundle to download: all | coco | vg
                     (default: all)
  --force            Re-download selected bundle(s)
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            if [[ $# -lt 2 ]]; then
                echo "--dest requires a value" >&2
                exit 1
            fi
            DEST_DIR="$2"
            shift 2
            ;;
        --subset)
            if [[ $# -lt 2 ]]; then
                echo "--subset requires a value (all|coco|vg)" >&2
                exit 1
            fi
            SUBSET="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${SUBSET}" != "all" && "${SUBSET}" != "coco" && "${SUBSET}" != "vg" ]]; then
    echo "Invalid --subset value: ${SUBSET}. Use all, coco, or vg." >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

if [[ "${FORCE}" -eq 1 ]]; then
    if [[ "${SUBSET}" == "all" || "${SUBSET}" == "coco" ]]; then
        rm -rf "${DEST_DIR}/gt-1000-coco"
    fi
    if [[ "${SUBSET}" == "all" || "${SUBSET}" == "vg" ]]; then
        rm -rf "${DEST_DIR}/gt-1000-vg"
    fi
fi

ALLOW_PATTERNS=""
if [[ "${SUBSET}" == "all" ]]; then
    ALLOW_PATTERNS="gt-1000-coco/**,gt-1000-vg/**"
elif [[ "${SUBSET}" == "coco" ]]; then
    ALLOW_PATTERNS="gt-1000-coco/**"
else
    ALLOW_PATTERNS="gt-1000-vg/**"
fi

echo "[xtc-dataset] Downloading ${SUBSET} bundle(s) from ${DATASET_REPO}"
echo "[xtc-dataset] Destination: ${DEST_DIR}"

# Conditionally enable hf_transfer if installed
USE_HF_TRANSFER=0
if python -c "import hf_transfer" &>/dev/null; then
    USE_HF_TRANSFER=1
fi

XTC_DATASET_REPO="${DATASET_REPO}" \
XTC_DEST_DIR="${DEST_DIR}" \
XTC_ALLOW_PATTERNS="${ALLOW_PATTERNS}" \
HF_HUB_ENABLE_HF_TRANSFER="${USE_HF_TRANSFER}" \
python - <<'PY'
import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

repo_id = os.environ["XTC_DATASET_REPO"]
local_dir = os.environ["XTC_DEST_DIR"]
allow_patterns = [p.strip() for p in os.environ["XTC_ALLOW_PATTERNS"].split(",") if p.strip()]

max_retries = 25
for attempt in range(max_retries):
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            max_workers=8, # Reduce concurrency slightly to avoid hitting limits too fast
        )
        print("[xtc-dataset] Download completed")
        break
    except Exception as e:
        # Check if it's a rate limit error (429)
        is_rate_limit = False
        if hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 429:
                is_rate_limit = True
        
        if is_rate_limit:
            print(f"\n[xtc-dataset] Rate limit hit (1000 req/5min). Waiting 5 minutes... (Attempt {attempt+1}/{max_retries})")
            time.sleep(305)
        else:
            raise e
PY

echo
echo "[xtc-dataset] Available GT bundles:"
[[ -d "${DEST_DIR}/gt-1000-coco" ]] && echo "  - ${DEST_DIR}/gt-1000-coco"
[[ -d "${DEST_DIR}/gt-1000-vg" ]] && echo "  - ${DEST_DIR}/gt-1000-vg"
