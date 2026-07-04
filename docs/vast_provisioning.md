# Vast.ai Provisioning

Use `scripts/provision_vast.sh` to set up SoulX-FlashTalk on a new Vast.ai instance.

The script:

- clones or fast-forwards `https://github.com/AhmadAFS1/SoulX-FlashTalk.git`
- creates a repo-local Python 3.10 venv at `/workspace/SoulX-FlashTalk/.venv`
- installs PyTorch `2.7.1` / torchvision `0.22.1` CUDA 12.8 wheels
- installs `requirements.txt`
- installs `flash_attn==2.8.0.post2`
- installs FFmpeg and common OpenCV runtime libraries when `apt-get` is available
- downloads both required Hugging Face model directories
- validates CUDA, imports, and `generate_video.py --help`

## Pasteable Vast Script

For a fresh instance, paste this into the Vast.ai provisioning script field:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace
cd /workspace

if [ ! -d SoulX-FlashTalk/.git ]; then
  git clone https://github.com/AhmadAFS1/SoulX-FlashTalk.git
fi

bash /workspace/SoulX-FlashTalk/scripts/provision_vast.sh
```

If the script has not been pushed to GitHub yet, paste the full contents of `scripts/provision_vast.sh` instead.

## Optional Environment Variables

```bash
REPO_URL=https://github.com/AhmadAFS1/SoulX-FlashTalk.git
REPO_BRANCH=main
WORKSPACE_DIR=/workspace
PYTHON_VERSION=3.10
RUN_SMOKE_TEST=0
```

Set `RUN_SMOKE_TEST=1` to run a short 2-second generation test after installation:

```bash
RUN_SMOKE_TEST=1 bash /workspace/SoulX-FlashTalk/scripts/provision_vast.sh
```

## After Provisioning

Activate the environment:

```bash
cd /workspace/SoulX-FlashTalk
source .venv/bin/activate
```

Run inference with CPU offload on GPUs below the no-offload VRAM threshold:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_video.py \
  --ckpt_dir models/SoulX-FlashTalk-14B \
  --wav2vec_dir models/chinese-wav2vec2-base \
  --cond_image examples/man.png \
  --audio_path examples/cantonese_16k.wav \
  --audio_encode_mode stream \
  --cpu_offload
```

## Notes

- The previous `huggingface-cli` command is deprecated in newer `huggingface_hub`; this script uses `hf download` when available.
- The setup is about 62 GB on disk after models and venv are installed.
- `pip check` may report `decord 0.6.0 is not supported on this platform`; in our tested environment the `decord` import still succeeds.
- If Hugging Face rate-limits anonymous downloads, set `HF_TOKEN` in the Vast.ai environment before provisioning.

## GB10 / aarch64 Install Differences

The GB10 environment is different from the RTX PRO 5000 Blackwell environment used for the benchmark in `docs/rtx_pro_5000_blackwell_benchmark.md`.

Observed GB10 setup:

```text
CPU architecture: aarch64
GPU: NVIDIA GB10
PyTorch: 2.7.1+cu128
PyTorch CUDA runtime: 12.8
Local CUDA toolkit: /usr/local/cuda-12.9
```

On this platform, several packages that are straightforward on the RTX PRO 5000 x86_64 environment do not have matching prebuilt wheels:

- `decord`
- `xformers==0.0.31`
- `flash_attn==2.8.0.post2`

The normal command can therefore fail before installing the rest of the dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Install the pure-Python and available wheel dependencies first, skipping the platform-sensitive packages:

```bash
.venv/bin/python -m pip install $(sed '/^decord\b/d;/^xformers\b/d' requirements.txt)
.venv/bin/python -m pip install omegaconf
```

`flash_attn` needs to compile from source when no matching prebuilt aarch64 wheel is available. On GB10/aarch64, use a low-parallelism build and only the local GPU architecture to avoid compiler processes being killed by memory pressure:

```bash
CUDA_HOME=/usr/local/cuda-12.9 \
PATH=/usr/local/cuda-12.9/bin:$PATH \
MAX_JOBS=1 \
NVCC_THREADS=1 \
FLASH_ATTN_CUDA_ARCHS=120 \
.venv/bin/python -m pip install flash_attn==2.8.0.post2 --no-build-isolation
```

The reduced build is slow because it compiles CUDA kernels locally. In the GB10 validation run it took about 90 minutes and produced a local wheel:

```text
flash_attn-2.8.0.post2-cp310-cp310-linux_aarch64.whl
```

A parallel build may start faster but can fail with `Killed` during `nvcc`/`cicc` compilation. The source build also warns that CUDA 12.9 is a minor-version mismatch with PyTorch's CUDA 12.8 runtime; this warning was not the immediate failure, but the toolkit/runtime mismatch is worth recording for reproducibility.

As of the GB10 validation run, these imports were fixed by the skip-and-install flow:

```text
imageio, librosa, loguru, transformers, diffusers, accelerate, gradio,
soundfile, cv2, einops, omegaconf, safetensors, huggingface_hub,
xfuser, optimum.quanto, flash_attn
```

Remaining GB10-specific caveats:

- `decord` has no matching pip wheel in this environment.
- `xformers==0.0.31` has no matching binary wheel and needs a compatible local source build or a repo fallback path.
