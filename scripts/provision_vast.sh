#!/usr/bin/env bash
# Provision SoulX-FlashTalk on a Vast.ai instance.
#
# This script is intentionally self-contained: it can be pasted directly into a
# Vast.ai provisioning script field, or run from an existing checkout.

set -Eeuo pipefail
IFS=$'\n\t'

REPO_URL="${REPO_URL:-https://github.com/AhmadAFS1/SoulX-FlashTalk.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_DIR}/SoulX-FlashTalk}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.0.post2}"
HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/.cache/huggingface}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_DIR}/.cache/pip}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"

export HF_HOME
export PIP_CACHE_DIR

log() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
    printf '\n[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
    printf '\n[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
    exit 1
}

on_error() {
    warn "Provisioning failed at line ${1}. Check the log above for the failing command."
}
trap 'on_error $LINENO' ERR

ensure_apt_packages() {
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not found; skipping system package installation."
        return
    fi

    if [ "$(id -u)" -ne 0 ]; then
        warn "Not running as root; skipping apt-get package installation."
        return
    fi

    log "Installing system packages..."
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        unzip
}

ensure_repo() {
    mkdir -p "${WORKSPACE_DIR}"

    if [ ! -d "${REPO_DIR}/.git" ]; then
        log "Cloning ${REPO_URL} into ${REPO_DIR}..."
        git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
        return
    fi

    log "Repository already exists at ${REPO_DIR}."
    cd "${REPO_DIR}"

    if git diff --quiet && git diff --cached --quiet; then
        log "Updating repository with git pull --ff-only..."
        git fetch origin "${REPO_BRANCH}"
        git checkout "${REPO_BRANCH}"
        git pull --ff-only origin "${REPO_BRANCH}"
    else
        warn "Repository has local changes; skipping git pull to avoid overwriting work."
    fi
}

check_disk_space() {
    local free_mb
    local free_gb

    free_mb="$(df -Pm "${WORKSPACE_DIR}" | awk 'NR==2 {print $4}')"
    free_gb="$((free_mb / 1024))"

    log "Disk available under ${WORKSPACE_DIR}: ${free_gb} GB."
    if [ "${free_gb}" -lt "${MIN_FREE_GB}" ]; then
        warn "Less than ${MIN_FREE_GB} GB free. A full install with models and venv uses about 62 GB; 80-100 GB free is recommended."
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"

    command -v uv >/dev/null 2>&1 || die "uv installation failed or uv is not on PATH."
}

venv_python_version() {
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        return 1
    fi
    "${VENV_DIR}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

ensure_venv() {
    ensure_uv

    local current_version=""
    current_version="$(venv_python_version 2>/dev/null || true)"

    if [ "${current_version}" != "${PYTHON_VERSION}" ]; then
        if [ -d "${VENV_DIR}" ]; then
            warn "Existing venv uses Python ${current_version:-unknown}; recreating ${VENV_DIR} with Python ${PYTHON_VERSION}."
            rm -rf "${VENV_DIR}"
        fi
        log "Installing Python ${PYTHON_VERSION} with uv..."
        uv python install "${PYTHON_VERSION}"
        log "Creating virtual environment at ${VENV_DIR}..."
        uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
    else
        log "Using existing Python ${PYTHON_VERSION} virtual environment at ${VENV_DIR}."
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    python -m ensurepip --upgrade >/dev/null 2>&1 || true
    python -m pip install --upgrade pip setuptools wheel packaging
}

install_python_dependencies() {
    cd "${REPO_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    log "Installing PyTorch ${TORCH_VERSION} and torchvision ${TORCHVISION_VERSION} from ${TORCH_INDEX_URL}..."
    pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

    log "Installing project requirements..."
    pip install -r requirements.txt

    log "Installing flash-attn ${FLASH_ATTN_VERSION}..."
    pip install ninja
    pip install "flash_attn==${FLASH_ATTN_VERSION}" --no-build-isolation

    log "Installing Hugging Face CLI..."
    pip install "huggingface_hub[cli,hf_transfer]"
}

hf_download() {
    local repo_id="$1"
    local dest="$2"
    shift 2
    local required_files=("$@")

    cd "${REPO_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    local complete=1
    for required in "${required_files[@]}"; do
        if [ ! -f "${dest}/${required}" ]; then
            complete=0
            break
        fi
    done

    if [ "${complete}" -eq 1 ]; then
        log "${repo_id} already appears complete at ${dest}; skipping download."
        return
    fi

    mkdir -p "${dest}"
    log "Downloading ${repo_id} into ${dest}..."

    if command -v hf >/dev/null 2>&1; then
        hf download "${repo_id}" --local-dir "${dest}"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        # Older images may still ship a working huggingface-cli.
        huggingface-cli download "${repo_id}" --local-dir "${dest}"
    else
        die "Neither hf nor huggingface-cli is available after installing huggingface_hub."
    fi
}

download_models() {
    mkdir -p "${REPO_DIR}/models"

    hf_download \
        "Soul-AILab/SoulX-FlashTalk-14B" \
        "${REPO_DIR}/models/SoulX-FlashTalk-14B" \
        "config.json" \
        "diffusion_pytorch_model.safetensors.index.json" \
        "diffusion_pytorch_model-00001-of-00004.safetensors" \
        "diffusion_pytorch_model-00002-of-00004.safetensors" \
        "diffusion_pytorch_model-00003-of-00004.safetensors" \
        "diffusion_pytorch_model-00004-of-00004.safetensors" \
        "Wan2.1_VAE.pth" \
        "models_t5_umt5-xxl-enc-bf16.pth" \
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        "xlm-roberta-large/tokenizer.json"

    hf_download \
        "TencentGameMate/chinese-wav2vec2-base" \
        "${REPO_DIR}/models/chinese-wav2vec2-base" \
        "config.json" \
        "preprocessor_config.json" \
        "pytorch_model.bin"
}

validate_install() {
    cd "${REPO_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    log "Validating Python imports and CUDA visibility..."
    python - <<'PY'
import importlib
import torch

modules = [
    "torch",
    "torchvision",
    "flash_attn",
    "cv2",
    "diffusers",
    "transformers",
    "accelerate",
    "gradio",
    "xfuser",
    "xformers",
    "librosa",
    "decord",
    "optimum.quanto",
]

print("python import validation")
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))

failed = []
for name in modules:
    try:
        mod = importlib.import_module(name)
        print("OK", name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print("FAIL", name, type(exc).__name__, exc)
        failed.append(name)

if failed:
    raise SystemExit(f"Import validation failed for: {failed}")
PY

    log "Running generate_video.py --help import smoke test..."
    python generate_video.py --help >/dev/null

    log "Running pip check for informational diagnostics..."
    if ! python -m pip check; then
        warn "pip check reported issues. If the only issue is 'decord 0.6.0 is not supported on this platform', it is known to import successfully in this environment."
    fi
}

run_optional_smoke_test() {
    if [ "${RUN_SMOKE_TEST}" != "1" ]; then
        return
    fi

    cd "${REPO_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    log "RUN_SMOKE_TEST=1; running a short generation smoke test with CPU offload..."
    mkdir -p sample_results
    ffmpeg -y -i examples/cantonese_16k.wav -t 2 -ar 16000 -ac 1 sample_results/provision_smoke_2s.wav >/tmp/flashtalk_provision_ffmpeg.log 2>&1

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python generate_video.py \
        --ckpt_dir models/SoulX-FlashTalk-14B \
        --wav2vec_dir models/chinese-wav2vec2-base \
        --input_prompt "A person is talking. Only the foreground character is moving, the background remains static." \
        --cond_image examples/man.png \
        --audio_path sample_results/provision_smoke_2s.wav \
        --audio_encode_mode stream \
        --cpu_offload \
        --save_file sample_results/provision_smoke.mp4
}

main() {
    log "Starting SoulX-FlashTalk Vast.ai provisioning."

    if command -v nvidia-smi >/dev/null 2>&1; then
        log "GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    else
        warn "nvidia-smi not found. GPU validation will happen through PyTorch later."
    fi

    ensure_apt_packages
    ensure_repo
    check_disk_space
    cd "${REPO_DIR}"
    ensure_venv
    install_python_dependencies
    download_models
    validate_install
    run_optional_smoke_test

    log "SoulX-FlashTalk setup complete."
    log "Activate with: source ${VENV_DIR}/bin/activate"
    log "Repo path: ${REPO_DIR}"
    log "Example run: CUDA_VISIBLE_DEVICES=0 python generate_video.py --ckpt_dir models/SoulX-FlashTalk-14B --wav2vec_dir models/chinese-wav2vec2-base --cond_image examples/man.png --audio_path examples/cantonese_16k.wav --audio_encode_mode stream --cpu_offload"
}

main "$@"
