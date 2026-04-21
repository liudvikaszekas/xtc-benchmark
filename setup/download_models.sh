#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KMAX_WEIGHTS_PATH="${REPO_ROOT}/weights/kmax_convnext_large.pth"
KMAX_GDRIVE_URL_PRIMARY="https://drive.usercontent.google.com/download?id=1b6rEnKw4PNTdqSdWpmb0P9dsvN0pkOiN&export=download"
KMAX_GDRIVE_URL_FALLBACK="https://drive.google.com/uc?export=download&id=1b6rEnKw4PNTdqSdWpmb0P9dsvN0pkOiN"

FAIR_PSG_MODEL_DIR="${REPO_ROOT}/full_workflow/models/masks-loc-sem"
FAIR_PSG_BEST_STATE_PATH="${FAIR_PSG_MODEL_DIR}/best_state.pth"
FAIR_PSG_CONFIG_PATH="${FAIR_PSG_MODEL_DIR}/config.json"
FAIR_PSG_HF_BEST_STATE_URL="https://huggingface.co/XTC-Bench/fair-psg-custom/resolve/main/best_state.pth?download=true"
FAIR_PSG_HF_CONFIG_URL="https://huggingface.co/XTC-Bench/fair-psg-custom/resolve/main/config.json?download=true"

BENCHMARK_WEIGHTS_MODELS_DIR="${REPO_ROOT}/weights/models"
BENCHMARK_MASKS_LOC_SEM_DIR="${BENCHMARK_WEIGHTS_MODELS_DIR}/masks-loc-sem"

KMAX_MIN_BYTES=100000000
FAIR_PSG_MIN_BYTES=200000000
FORCE=0

usage() {
    cat <<'EOF'
Usage: setup/download_models.sh [options]

Downloads required benchmark model artifacts:
- weights/kmax_convnext_large.pth
- full_workflow/models/masks-loc-sem/best_state.pth
- full_workflow/models/masks-loc-sem/config.json

Also creates compatibility symlink:
- weights/models/masks-loc-sem -> ../../../full_workflow/models/masks-loc-sem

Options:
  --force      Re-download files even if they already exist and look valid.
  -h, --help   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

download_file() {
    local destination="$1"
    shift

    local parent
    parent="$(dirname "${destination}")"
    mkdir -p "${parent}"

    local tmp_file
    tmp_file="${destination}.tmp"

    local url
    for url in "$@"; do
        echo "[download_models] Download attempt: ${url}"
        if curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 -o "${tmp_file}" "${url}"; then
            if [[ -s "${tmp_file}" ]]; then
                mv "${tmp_file}" "${destination}"
                return 0
            fi
        fi
    done

    rm -f "${tmp_file}"
    return 1
}

download_from_gdrive() {
    local file_id="$1"
    local destination="$2"

    local parent
    parent="$(dirname "${destination}")"
    mkdir -p "${parent}"

    local tmp_file
    local cookie_file
    tmp_file="${destination}.tmp"
    cookie_file="${destination}.cookie"

    local base_url
    base_url="https://drive.google.com/uc?export=download&id=${file_id}"

    echo "[download_models] Google Drive attempt: ${base_url}"
    if ! curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 \
        -c "${cookie_file}" -o "${tmp_file}" "${base_url}"; then
        rm -f "${tmp_file}" "${cookie_file}"
        return 1
    fi

    if grep -qi "<html" "${tmp_file}"; then
        if grep -qi "virus scan warning" "${tmp_file}"; then
            local form_action
            local confirm
            local uuid
            local export_flag
            local parsed_id

            form_action="$(grep -o 'action="[^"]*"' "${tmp_file}" | head -n1 | sed 's/^action="//; s/"$//' || true)"
            confirm="$(grep -o 'name="confirm" value="[^"]*"' "${tmp_file}" | head -n1 | sed 's/.*value="//; s/"$//' || true)"
            uuid="$(grep -o 'name="uuid" value="[^"]*"' "${tmp_file}" | head -n1 | sed 's/.*value="//; s/"$//' || true)"
            export_flag="$(grep -o 'name="export" value="[^"]*"' "${tmp_file}" | head -n1 | sed 's/.*value="//; s/"$//' || true)"
            parsed_id="$(grep -o 'name="id" value="[^"]*"' "${tmp_file}" | head -n1 | sed 's/.*value="//; s/"$//' || true)"

            if [[ -n "${form_action}" && "${form_action}" != http* ]]; then
                form_action="https://drive.google.com${form_action}"
            fi

            if [[ -z "${form_action}" || -z "${confirm}" || -z "${parsed_id}" ]]; then
                rm -f "${tmp_file}" "${cookie_file}"
                return 1
            fi

            echo "[download_models] Google Drive confirm step"
            local -a curl_cmd
            curl_cmd=(
                curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20
                -b "${cookie_file}"
                --get
                --data-urlencode "id=${parsed_id}"
                --data-urlencode "export=${export_flag:-download}"
                --data-urlencode "confirm=${confirm}"
            )

            if [[ -n "${uuid}" ]]; then
                curl_cmd+=(--data-urlencode "uuid=${uuid}")
            fi

            curl_cmd+=(-o "${tmp_file}" "${form_action}")

            if ! "${curl_cmd[@]}"; then
                rm -f "${tmp_file}" "${cookie_file}"
                return 1
            fi
        else
            echo "[download_models] Google Drive returned HTML page that is not a download confirmation page." >&2
            rm -f "${tmp_file}" "${cookie_file}"
            return 1
        fi
    fi

    if grep -qi "<html" "${tmp_file}"; then
        echo "[download_models] Google Drive still returned HTML instead of binary content." >&2
        rm -f "${tmp_file}" "${cookie_file}"
        return 1
    fi

    mv "${tmp_file}" "${destination}"
    rm -f "${cookie_file}"
    return 0
}

ensure_min_size() {
    local path="$1"
    local min_bytes="$2"
    if [[ ! -f "${path}" ]]; then
        return 1
    fi

    local file_size
    file_size="$(wc -c < "${path}")"
    if [[ "${file_size}" -lt "${min_bytes}" ]]; then
        return 1
    fi
    return 0
}

mkdir -p "${REPO_ROOT}/weights"
mkdir -p "${FAIR_PSG_MODEL_DIR}"

if [[ "${FORCE}" -eq 0 ]] && ensure_min_size "${KMAX_WEIGHTS_PATH}" "${KMAX_MIN_BYTES}"; then
    echo "[download_models] Reusing existing kMaX weights at ${KMAX_WEIGHTS_PATH}"
else
    echo "[download_models] Downloading kMaX weights"
    if download_from_gdrive "1b6rEnKw4PNTdqSdWpmb0P9dsvN0pkOiN" "${KMAX_WEIGHTS_PATH}" || \
       download_file "${KMAX_WEIGHTS_PATH}" "${KMAX_GDRIVE_URL_PRIMARY}" "${KMAX_GDRIVE_URL_FALLBACK}"; then
        if ! ensure_min_size "${KMAX_WEIGHTS_PATH}" "${KMAX_MIN_BYTES}"; then
            echo "[download_models] Downloaded kMaX file seems too small/corrupt: ${KMAX_WEIGHTS_PATH}" >&2
            exit 1
        fi
    else
        echo "[download_models] Failed to download kMaX weights from Google Drive URLs." >&2
        exit 1
    fi
fi

if [[ "${FORCE}" -eq 0 ]] && ensure_min_size "${FAIR_PSG_BEST_STATE_PATH}" "${FAIR_PSG_MIN_BYTES}"; then
    echo "[download_models] Reusing existing fair-psg best_state at ${FAIR_PSG_BEST_STATE_PATH}"
else
    echo "[download_models] Downloading fair-psg best_state.pth"
    if download_file "${FAIR_PSG_BEST_STATE_PATH}" "${FAIR_PSG_HF_BEST_STATE_URL}"; then
        if ! ensure_min_size "${FAIR_PSG_BEST_STATE_PATH}" "${FAIR_PSG_MIN_BYTES}"; then
            echo "[download_models] Downloaded fair-psg checkpoint seems too small/corrupt: ${FAIR_PSG_BEST_STATE_PATH}" >&2
            exit 1
        fi
    else
        echo "[download_models] Failed to download fair-psg best_state.pth from Hugging Face." >&2
        exit 1
    fi
fi

if [[ "${FORCE}" -eq 0 ]] && [[ -f "${FAIR_PSG_CONFIG_PATH}" ]]; then
    echo "[download_models] Reusing existing fair-psg config at ${FAIR_PSG_CONFIG_PATH}"
else
    echo "[download_models] Downloading fair-psg config.json"
    if ! download_file "${FAIR_PSG_CONFIG_PATH}" "${FAIR_PSG_HF_CONFIG_URL}"; then
        echo "[download_models] Failed to download fair-psg config.json from Hugging Face." >&2
        echo "[download_models] Scene graph generation requires both best_state.pth and config.json in ${FAIR_PSG_MODEL_DIR}" >&2
        exit 1
    fi
fi

mkdir -p "${BENCHMARK_WEIGHTS_MODELS_DIR}"
if [[ -L "${BENCHMARK_MASKS_LOC_SEM_DIR}" ]]; then
    echo "[download_models] Replacing symlink with standalone directory at ${BENCHMARK_MASKS_LOC_SEM_DIR}"
    rm -f "${BENCHMARK_MASKS_LOC_SEM_DIR}"
fi

if [[ -e "${BENCHMARK_MASKS_LOC_SEM_DIR}" && ! -d "${BENCHMARK_MASKS_LOC_SEM_DIR}" ]]; then
    echo "[download_models] Target exists but is not a directory: ${BENCHMARK_MASKS_LOC_SEM_DIR}" >&2
    exit 1
fi

mkdir -p "${BENCHMARK_MASKS_LOC_SEM_DIR}"
cp -a "${FAIR_PSG_MODEL_DIR}/." "${BENCHMARK_MASKS_LOC_SEM_DIR}/"
echo "[download_models] Synced standalone model directory: ${BENCHMARK_MASKS_LOC_SEM_DIR}"

echo

echo "[download_models] Completed successfully."
echo "[download_models] kMaX weights: ${KMAX_WEIGHTS_PATH}"
echo "[download_models] fair-psg model dir: ${FAIR_PSG_MODEL_DIR}"
