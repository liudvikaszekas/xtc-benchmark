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

XTC_DATASET_REPO="${DATASET_REPO}" \
XTC_DEST_DIR="${DEST_DIR}" \
XTC_ALLOW_PATTERNS="${ALLOW_PATTERNS}" \
python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["XTC_DATASET_REPO"]
local_dir = os.environ["XTC_DEST_DIR"]
allow_patterns = [p.strip() for p in os.environ["XTC_ALLOW_PATTERNS"].split(",") if p.strip()]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=allow_patterns,
)

print("[xtc-dataset] Download completed")
print(f"[xtc-dataset] Local path: {local_dir}")
PY

echo
echo "[xtc-dataset] Available GT bundles:"
[[ -d "${DEST_DIR}/gt-1000-coco" ]] && echo "  - ${DEST_DIR}/gt-1000-coco"
[[ -d "${DEST_DIR}/gt-1000-vg" ]] && echo "  - ${DEST_DIR}/gt-1000-vg"
