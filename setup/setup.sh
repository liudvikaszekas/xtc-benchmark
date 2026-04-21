#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TMP_WORK_DIR="${REPO_ROOT}/.tmp"
TMP_DISTRO_DIR="${TMP_WORK_DIR}/distros"
ENV_YAML_DIR="${REPO_ROOT}/benchmark/setup/environments"
MODEL_DOWNLOAD_SCRIPT="${REPO_ROOT}/benchmark/setup/download_models.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

RUN_SMOKE=1
USE_BLACKWELL=0
ONLY_ENVS=""
RUN_TAG=""
RECREATE=0
EXTERNAL_UNIVLM_PATH=""
TORCH_VARIANT="auto"
SKIP_WEIGHT_DOWNLOAD=0
INSTALL_TARGET="temporary"
EXISTING_CONDA_ROOT=""

TEMP_CONDA_ROOT=""
MINICONDA_DIR=""
ENV_ROOT=""
PKGS_ROOT=""
PIP_CACHE_DIR=""
REPORT_PATH=""
CONFIG_SNIPPET_PATH=""
MINICONDA_INSTALLER=""
CONDA_EXE=""

usage() {
    cat <<'EOF'
Usage: benchmark/setup/setup.sh [options]

Creates benchmark conda environments from portable YAML files (no lockfiles)
in either:
    - a fresh isolated Miniconda distro under .tmp/distros/<tag>/ (default), or
    - an existing conda installation (advanced).

Options:
  --run-tag TAG       Custom tag for .tmp/distros/<tag>. Defaults to timestamp_pid.
  --only a,b,c        Setup only selected env names.
    --install-target T  Conda install target: temporary or existing (default: temporary).
    --existing-conda-root PATH
                                            Existing conda base path (required for --install-target existing
                                            unless auto-detect succeeds via 'conda info --base').
  --skip-smoke        Skip smoke import checks.
  --blackwell         Use generate_any_scene/environment_blackwell.yml update.
  --recreate          If run tag exists, remove it before setup.
    --skip-weights      Skip automatic model weight download.
  --with-external-univlm PATH
                      Optional external univlm checkout path to init submodules.
    --torch-variant V   Torch install variant: auto, cpu, cu121 (default: auto).
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-tag)
            if [[ $# -lt 2 ]]; then
                echo "--run-tag requires a value" >&2
                exit 1
            fi
            RUN_TAG="$2"
            shift 2
            ;;
        --only)
            if [[ $# -lt 2 ]]; then
                echo "--only requires a comma-separated env list" >&2
                exit 1
            fi
            ONLY_ENVS="$2"
            shift 2
            ;;
        --install-target)
            if [[ $# -lt 2 ]]; then
                echo "--install-target requires a value (temporary|existing)" >&2
                exit 1
            fi
            INSTALL_TARGET="$2"
            shift 2
            ;;
        --existing-conda-root)
            if [[ $# -lt 2 ]]; then
                echo "--existing-conda-root requires a path" >&2
                exit 1
            fi
            EXISTING_CONDA_ROOT="$2"
            shift 2
            ;;
        --skip-smoke)
            RUN_SMOKE=0
            shift
            ;;
        --blackwell)
            USE_BLACKWELL=1
            shift
            ;;
        --recreate)
            RECREATE=1
            shift
            ;;
        --skip-weights)
            SKIP_WEIGHT_DOWNLOAD=1
            shift
            ;;
        --with-external-univlm)
            if [[ $# -lt 2 ]]; then
                echo "--with-external-univlm requires a path" >&2
                exit 1
            fi
            EXTERNAL_UNIVLM_PATH="$2"
            shift 2
            ;;
        --torch-variant)
            if [[ $# -lt 2 ]]; then
                echo "--torch-variant requires a value (auto|cpu|cu121)" >&2
                exit 1
            fi
            TORCH_VARIANT="$2"
            shift 2
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

if [[ "${INSTALL_TARGET}" != "temporary" && "${INSTALL_TARGET}" != "existing" ]]; then
    echo "Invalid --install-target value: ${INSTALL_TARGET}. Use temporary or existing." >&2
    exit 1
fi

if [[ "${INSTALL_TARGET}" == "temporary" && -n "${EXISTING_CONDA_ROOT}" ]]; then
    echo "--existing-conda-root can only be used with --install-target existing." >&2
    exit 1
fi

if [[ "${TORCH_VARIANT}" != "auto" && "${TORCH_VARIANT}" != "cpu" && "${TORCH_VARIANT}" != "cu121" ]]; then
    echo "Invalid --torch-variant value: ${TORCH_VARIANT}. Use auto, cpu, or cu121." >&2
    exit 1
fi

if [[ -z "${RUN_TAG}" ]]; then
    RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
fi

TEMP_CONDA_ROOT="${TMP_DISTRO_DIR}/${RUN_TAG}"
MINICONDA_DIR=""
ENV_ROOT=""
PKGS_ROOT="${TEMP_CONDA_ROOT}/pkgs"
PIP_CACHE_DIR="${TEMP_CONDA_ROOT}/pip_cache"
REPORT_PATH="${TEMP_CONDA_ROOT}/setup_report_v2.txt"
CONFIG_SNIPPET_PATH="${TEMP_CONDA_ROOT}/config_conda_init_snippet.json"
MINICONDA_INSTALLER="${TEMP_CONDA_ROOT}/Miniconda3-latest-Linux-x86_64.sh"

mkdir -p "${TMP_WORK_DIR}" "${TMP_DISTRO_DIR}"

if [[ "${RECREATE}" -eq 1 && -d "${TEMP_CONDA_ROOT}" ]]; then
    echo "[setup_v2] Recreate requested. Removing ${TEMP_CONDA_ROOT}"
    rm -rf "${TEMP_CONDA_ROOT}"
fi

mkdir -p "${TEMP_CONDA_ROOT}" "${PKGS_ROOT}" "${PIP_CACHE_DIR}"

if [[ "${INSTALL_TARGET}" == "temporary" ]]; then
    MINICONDA_DIR="${TEMP_CONDA_ROOT}/miniconda3"
    ENV_ROOT="${TEMP_CONDA_ROOT}/envs"
else
    if [[ -z "${EXISTING_CONDA_ROOT}" ]]; then
        if command -v conda >/dev/null 2>&1; then
            EXISTING_CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
        fi
    fi

    if [[ -z "${EXISTING_CONDA_ROOT}" ]]; then
        echo "Could not determine existing conda root automatically." >&2
        echo "Please provide --existing-conda-root PATH." >&2
        exit 1
    fi

    MINICONDA_DIR="${EXISTING_CONDA_ROOT}"
    ENV_ROOT="${EXISTING_CONDA_ROOT}/envs/xtc-bench-${RUN_TAG}"
fi

if [[ "${INSTALL_TARGET}" == "existing" && "${RECREATE}" -eq 1 && -d "${ENV_ROOT}" ]]; then
    echo "[setup_v2] Recreate requested. Removing existing-target env root ${ENV_ROOT}"
    rm -rf "${ENV_ROOT}"
fi

mkdir -p "${ENV_ROOT}" "${PKGS_ROOT}" "${PIP_CACHE_DIR}"

if [[ "${TEMP_CONDA_ROOT}" != "${REPO_ROOT}"/* ]]; then
    echo "Refusing to use conda root outside repository root: ${TEMP_CONDA_ROOT}" >&2
    exit 1
fi

RUN_TMPDIR="${TEMP_CONDA_ROOT}/tmp_work"
mkdir -p "${RUN_TMPDIR}"
export TMPDIR="${RUN_TMPDIR}"
export CONDA_PKGS_DIRS="${PKGS_ROOT}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

record_report() {
    echo "$1" | tee -a "${REPORT_PATH}"
}

printf "XTC-Bench setup v2 report\n" > "${REPORT_PATH}"
printf "timestamp=%s\n" "$(date -Iseconds)" >> "${REPORT_PATH}"
printf "repo_root=%s\n" "${REPO_ROOT}" >> "${REPORT_PATH}"
printf "install_target=%s\n" "${INSTALL_TARGET}" >> "${REPORT_PATH}"
printf "torch_variant=%s\n" "${TORCH_VARIANT}" >> "${REPORT_PATH}"
printf "run_tag=%s\n" "${RUN_TAG}" >> "${REPORT_PATH}"
printf "temp_conda_root=%s\n\n" "${TEMP_CONDA_ROOT}" >> "${REPORT_PATH}"

if [[ -n "${ONLY_ENVS}" ]]; then
    record_report "warning: partial_install=only_envs:${ONLY_ENVS}"
fi

bootstrap_miniconda() {
    if [[ "${INSTALL_TARGET}" == "existing" ]]; then
        if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
            echo "Existing conda root does not contain conda executable: ${MINICONDA_DIR}/bin/conda" >&2
            exit 1
        fi
        echo "[setup_v2] Using existing conda root at ${MINICONDA_DIR}"
        return
    fi

    if [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
        echo "[setup_v2] Reusing local Miniconda at ${MINICONDA_DIR}"
        return
    fi

    echo "[setup_v2] Downloading Miniconda installer"
    curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"

    echo "[setup_v2] Installing Miniconda into ${MINICONDA_DIR}"
    bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_DIR}"
}

bootstrap_miniconda
CONDA_EXE="${MINICONDA_DIR}/bin/conda"
record_report "conda: install_target=${INSTALL_TARGET} root=${MINICONDA_DIR} exe=${CONDA_EXE}"
record_report "env_root: ${ENV_ROOT}"

if [[ "${SKIP_WEIGHT_DOWNLOAD}" -eq 1 ]]; then
    echo "[setup_v2] Skipping automatic weight downloads (--skip-weights)."
    record_report "weights: skipped=true"
else
    if [[ ! -f "${MODEL_DOWNLOAD_SCRIPT}" ]]; then
        echo "[setup_v2] Missing model download script: ${MODEL_DOWNLOAD_SCRIPT}" >&2
        exit 1
    fi

    echo "[setup_v2] Downloading required model weights via ${MODEL_DOWNLOAD_SCRIPT}"
    bash "${MODEL_DOWNLOAD_SCRIPT}"
    record_report "weights: download_script=ok script=${MODEL_DOWNLOAD_SCRIPT}"
fi

env_prefix() {
    local env_name="$1"
    echo "${ENV_ROOT}/${env_name}"
}

env_python() {
    local env_name="$1"
    echo "$(env_prefix "${env_name}")/bin/python"
}

ensure_env_pip() {
    local env_name="$1"
    local python_bin
    python_bin="$(env_python "${env_name}")"
    if ! "${python_bin}" -m pip --version >/dev/null 2>&1; then
        echo "[setup_v2] Bootstrapping pip in ${env_name}"
        "${python_bin}" -m ensurepip --upgrade
    fi
}

conda_pip_install() {
    local env_name="$1"
    shift
    local python_bin
    python_bin="$(env_python "${env_name}")"
    ensure_env_pip "${env_name}"
    env -u CC -u CXX "${python_bin}" -m pip install "$@"
}

conda_clean_env_cmd() {
    # CRITICAL: Pip/Setuptools in conda use build vars to choose compilers.
    # If these are set to cross-compilers or old paths, pip will fail.
    # We unset EVERYTHING relating to compilers and the current conda session.
    env \
        -u CC -u CXX \
        -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
        -u CMAKE_PREFIX_PATH -u CMAKE_ARGS -u CMAKE_LIBRARY_PATH -u CMAKE_INCLUDE_PATH \
        -u CONDA_BUILD_SYSROOT -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
        -u CONDA_EXE -u CONDA_PYTHON_EXE -u CONDA_SHLVL \
        -u PYTHONHOME -u PYTHONPATH \
        -u PKG_CONFIG_PATH \
        "$@"
}

try_pip_then_git_fallback() {
    local env_name="$1"
    local pypi_pkg="$2"
    local git_pkg="$3"

    set +e
    conda_pip_install "${env_name}" "${pypi_pkg}"
    local rc=$?
    set -e

    if [[ ${rc} -ne 0 ]]; then
        echo "[setup_v2] PyPI install failed for ${pypi_pkg}. Trying ${git_pkg}."
        conda_pip_install "${env_name}" "${git_pkg}"
    fi
}

install_kmax_detectron2() {
    local env_name="$1"
    local prefix
    local python_bin
    local arch_list

    prefix="$(env_prefix "${env_name}")"
    python_bin="$(env_python "${env_name}")"
    arch_list="${TORCH_CUDA_ARCH_LIST:-8.0}"

    echo "[setup_v2] Installing CUDA toolkit headers for detectron2 build in ${env_name}"
    "${CONDA_EXE}" install -y --prefix "${prefix}" -c nvidia -c conda-forge cuda-toolkit=12.1

    ensure_env_pip "${env_name}"
    env -u CC -u CXX \
        -u CMAKE_PREFIX_PATH -u CONDA_BUILD_SYSROOT -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
        -u CONDA_EXE -u CONDA_PYTHON_EXE -u CONDA_SHLVL \
        PATH="${prefix}/bin:${PATH}" \
        CUDA_HOME="${prefix}" \
        CPATH="${prefix}/include" \
        LD_LIBRARY_PATH="${prefix}/lib:${prefix}/lib64:${LD_LIBRARY_PATH:-}" \
        TORCH_CUDA_ARCH_LIST="${arch_list}" \
        FORCE_CUDA=1 \
        "${python_bin}" -m pip install --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git"
}

install_torch_stack() {
    local env_name="$1"
    local variant="$2"

    local desired_variant="${variant}"
    if [[ "${desired_variant}" == "auto" ]]; then
        desired_variant="cu121"
    fi

    if [[ "${desired_variant}" == "cpu" ]]; then
        conda_pip_install "${env_name}" --index-url https://download.pytorch.org/whl/cpu torch torchvision
        return 0
    fi

    if [[ "${desired_variant}" == "cu121" ]]; then
        set +e
        conda_pip_install "${env_name}" --index-url https://download.pytorch.org/whl/cu121 torch torchvision
        local cu_rc=$?
        set -e

        if [[ ${cu_rc} -eq 0 ]]; then
            return 0
        fi

        echo "[setup_v2] Torch cu121 install failed for ${env_name}, falling back to cpu wheels."
        conda_pip_install "${env_name}" --index-url https://download.pytorch.org/whl/cpu torch torchvision
        return 0
    fi

    echo "[setup_v2] Unknown torch variant requested for ${env_name}: ${variant}" >&2
    return 1
}

should_install_for_mode() {
    local env_name="$1"

    return 0
}

env_selected() {
    local env_name="$1"

    if ! should_install_for_mode "${env_name}"; then
        return 1
    fi

    if [[ -z "${ONLY_ENVS}" ]]; then
        return 0
    fi

    local token
    IFS="," read -ra tokens <<< "${ONLY_ENVS}"
    for token in "${tokens[@]}"; do
        if [[ "${token}" == "${env_name}" ]]; then
            return 0
        fi
    done

    return 1
}

create_env_from_yaml() {
    local env_name="$1"
    local env_file="$2"
    local prefix
    prefix="$(env_prefix "${env_name}")"

    if [[ ! -f "${env_file}" ]]; then
        echo "Missing env definition for ${env_name}: ${env_file}" >&2
        exit 1
    fi

    if [[ -d "${prefix}" ]]; then
        echo "[setup_v2] Env already exists, skipping create: ${env_name}"
        record_report "${env_name}: create=skipped reason=exists"
        return
    fi

    echo "[setup_v2] Creating ${env_name} from ${env_file}"
    conda_clean_env_cmd "${CONDA_EXE}" env create --prefix "${prefix}" -f "${env_file}"
    record_report "${env_name}: create=ok env_file=${env_file} prefix=${prefix}"
}

declare -a ENVS=(
    "kmax_env:${ENV_YAML_DIR}/kmax_env.yml"
    "fair-psg:${ENV_YAML_DIR}/fair-psg_env.yml"
    "vllm_env:${ENV_YAML_DIR}/vllm_env.yml"
    "univlm:${ENV_YAML_DIR}/univlm_env.yml"
    "GenerateAnyScene:${ENV_YAML_DIR}/GenerateAnyScene_env.yml"
    "graph_matching:${ENV_YAML_DIR}/graph_matching_env.yml"
)

for item in "${ENVS[@]}"; do
    IFS=":" read -r env_name env_file <<< "${item}"
    if env_selected "${env_name}"; then
        create_env_from_yaml "${env_name}" "${env_file}"
    else
        record_report "${env_name}: create=skipped only=${ONLY_ENVS}"
    fi
done

echo "[setup_v2] Ensuring benchmark submodules are initialized"
git -C "${REPO_ROOT}" submodule update --init --recursive


if env_selected "fair-psg"; then
    FAIR_SUBMODULE="${REPO_ROOT}/benchmark/submodules/panoptic-scene-graph-generation"
    if [[ -d "${FAIR_SUBMODULE}" ]]; then
        install_torch_stack "fair-psg" "${TORCH_VARIANT}"
        record_report "fair-psg: torch_install=ok variant=${TORCH_VARIANT}"

        echo "[setup_v2] Installing fair-psg runtime extras"
        conda_pip_install "fair-psg" tensorboard scikit-learn timm
        record_report "fair-psg: runtime_extras_install=ok packages=tensorboard,scikit-learn,timm"

        echo "[setup_v2] Installing fair-psg submodule in editable mode"
        ensure_env_pip "fair-psg"

        set +e
        env -u CC -u CXX "$(env_python fair-psg)" -m pip install -e "${FAIR_SUBMODULE}"
        fair_editable_rc=$?
        set -e

        if [[ ${fair_editable_rc} -eq 0 ]]; then
            record_report "fair-psg: editable_submodule_install=ok path=${FAIR_SUBMODULE}"
        else
            echo "[setup_v2] fair-psg editable install failed, applying .pth fallback"
            "$(env_python fair-psg)" - <<PY
import site
from pathlib import Path

target = Path(r"${FAIR_SUBMODULE}").resolve()
site_pkgs = [Path(p) for p in site.getsitepackages() if p.endswith("site-packages")]
if not site_pkgs:
    raise SystemExit("No site-packages directory found for fair-psg")
pth = site_pkgs[0] / "fair_psgg_local_path.pth"
pth.write_text(str(target) + "\n", encoding="utf-8")
print(f"Wrote {pth}")
PY
            record_report "fair-psg: editable_submodule_install=fallback_pth path=${FAIR_SUBMODULE}"
        fi

        set +e
        try_pip_then_git_fallback "fair-psg" "panopticapi" "git+https://github.com/cocodataset/panopticapi.git"
        panoptic_rc=$?
        set -e
        if [[ ${panoptic_rc} -eq 0 ]]; then
            record_report "fair-psg: panopticapi_install=ok"
        else
            record_report "fair-psg: panopticapi_install=failed"
        fi
    else
        echo "[setup_v2] Missing fair-psg submodule path: ${FAIR_SUBMODULE}" >&2
        record_report "fair-psg: editable_submodule_install=missing path=${FAIR_SUBMODULE}"
    fi
fi

if env_selected "vllm_env"; then
    echo "[setup_v2] Installing vLLM runtime packages"
    conda_pip_install "vllm_env" vllm pydantic numpy pandas pillow tqdm "transformers<5" inflect sentence-transformers
    record_report "vllm_env: runtime_install=ok"
fi

if env_selected "GenerateAnyScene"; then
    install_torch_stack "GenerateAnyScene" "${TORCH_VARIANT}"
    record_report "GenerateAnyScene: torch_install=ok variant=${TORCH_VARIANT}"

    GAS_PREFIX="$(env_prefix GenerateAnyScene)"
    GAS_ENV_FILE="${REPO_ROOT}/benchmark/scripts/pipeline/generate_any_scene/environment.yml"
    GAS_ENV_FILE_BLACKWELL="${REPO_ROOT}/benchmark/scripts/pipeline/generate_any_scene/environment_blackwell.yml"
    if [[ "${USE_BLACKWELL}" -eq 1 ]]; then
        GAS_ENV_FILE="${GAS_ENV_FILE_BLACKWELL}"
    fi

    if [[ -f "${GAS_ENV_FILE}" ]]; then
        echo "[setup_v2] Updating GenerateAnyScene from ${GAS_ENV_FILE}"
        conda_clean_env_cmd "${CONDA_EXE}" env update --prefix "${GAS_PREFIX}" -f "${GAS_ENV_FILE}" --prune
        record_report "GenerateAnyScene: env_update=ok source=${GAS_ENV_FILE}"
    else
        record_report "GenerateAnyScene: env_update=missing source=${GAS_ENV_FILE}"
    fi
fi

if env_selected "graph_matching"; then
    echo "[setup_v2] Installing graph_matching runtime packages"
    install_torch_stack "graph_matching" "cpu"
    conda_pip_install "graph_matching" transformers sentence-transformers
    record_report "graph_matching: runtime_install=ok"
fi

if env_selected "univlm"; then
    echo "[setup_v2] Installing univlm extra dependencies"
    conda_pip_install "univlm" tiktoken typeguard
    install_torch_stack "univlm" "${TORCH_VARIANT}"
    record_report "univlm: torch_install=ok variant=${TORCH_VARIANT}"

    if try_pip_then_git_fallback "univlm" "flash-attn" "git+https://github.com/Dao-AILab/flash-attention.git"; then
        record_report "univlm: flash_attn=installed"
    else
        record_report "univlm: flash_attn=failed_optional"
    fi
fi

if env_selected "kmax_env"; then
    echo "[setup_v2] Installing kmax extras"
    install_torch_stack "kmax_env" "${TORCH_VARIANT}"
    record_report "kmax_env: torch_install=ok variant=${TORCH_VARIANT}"

    conda_pip_install "kmax_env" opencv-python-headless
    conda_pip_install "kmax_env" timm
    set +e
    install_kmax_detectron2 "kmax_env"
    kmax_rc=$?
    set -e
    if [[ ${kmax_rc} -eq 0 ]]; then
        record_report "kmax_env: detectron2_install=ok method=git-build-with-cuda-toolkit"
    else
        record_report "kmax_env: detectron2_install=failed"
    fi
fi

if [[ -z "${EXTERNAL_UNIVLM_PATH}" ]]; then
    EXTERNAL_UNIVLM_PATH="$(cd "${REPO_ROOT}/.." && pwd)/univlm"
fi
if env_selected "univlm"; then
    if [[ -d "${EXTERNAL_UNIVLM_PATH}" ]]; then
        git -C "${EXTERNAL_UNIVLM_PATH}" submodule update --init --recursive || true
        record_report "univlm: external_repo_found path=${EXTERNAL_UNIVLM_PATH}"
    else
        record_report "univlm: external_repo_missing path=${EXTERNAL_UNIVLM_PATH}"
    fi
fi

write_conda_init_helper() {
    local helper_path="${TEMP_CONDA_ROOT}/conda_init.sh"
    cat > "${helper_path}" <<EOF
#!/usr/bin/env bash
set -e
if [[ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
else
    source "${MINICONDA_DIR}/bin/activate"
fi
export CONDA_ENVS_PATH="${ENV_ROOT}"
EOF
    chmod +x "${helper_path}"
    record_report "helper: conda_init_script=${helper_path}"
}

write_config_snippet() {
        cat > "${CONFIG_SNIPPET_PATH}" <<EOF
{
    "conda_init_script": "source ${TEMP_CONDA_ROOT}/conda_init.sh"
}
EOF
        record_report "helper: config_snippet=${CONFIG_SNIPPET_PATH}"
}

smoke_check() {
    local env_name="$1"
    local code="$2"
    local python_bin
    python_bin="$(env_python "${env_name}")"

    set +e
    "${python_bin}" -c "${code}"
    local rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then
        record_report "${env_name}: smoke_check=ok"
    else
        record_report "${env_name}: smoke_check=failed"
        return 1
    fi
}

write_conda_init_helper
write_config_snippet

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
    echo "[setup_v2] Running smoke checks"
    env_selected "kmax_env" && smoke_check "kmax_env" "import torch, numpy, PIL, cv2"
    env_selected "fair-psg" && smoke_check "fair-psg" "import torch, numpy, transformers, sklearn, tensorboard, timm; from fair_psgg.tasks.inference import inference2"
    env_selected "vllm_env" && smoke_check "vllm_env" "import vllm, pydantic, numpy, pandas, PIL, tqdm, transformers, inflect, sentence_transformers"
    env_selected "GenerateAnyScene" && smoke_check "GenerateAnyScene" "import networkx, pandas, matplotlib, pydantic, inflect, sentence_transformers"
    env_selected "univlm" && smoke_check "univlm" "import torch, tiktoken, typeguard"
    env_selected "graph_matching" && smoke_check "graph_matching" "import torch, scipy, networkx, transformers, sentence_transformers"
fi

echo
echo "[setup_v2] Completed successfully."
echo "[setup_v2] Install target: ${INSTALL_TARGET}"
echo "[setup_v2] Conda root: ${MINICONDA_DIR}"
echo "[setup_v2] Local conda root: ${TEMP_CONDA_ROOT}"
echo "[setup_v2] Setup report: ${REPORT_PATH}"
echo "[setup_v2] Config snippet: ${CONFIG_SNIPPET_PATH}"
echo "[setup_v2] Use this conda init script in benchmark config: ${TEMP_CONDA_ROOT}/conda_init.sh"
