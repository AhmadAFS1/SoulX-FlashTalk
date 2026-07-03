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
