# RTX PRO 5000 Blackwell Benchmark

Date: 2026-07-03

This file records the current GPU baseline for SoulX-FlashTalk so a future 1x GB10 run can be compared against the same commands and outputs.

## Hardware

```text
GPU: NVIDIA RTX PRO 5000 Blackwell
Architecture: Blackwell
Driver: 595.71.05
CUDA reported by driver: 13.2
PyTorch: 2.7.1+cu128
PyTorch CUDA runtime: 12.8
VRAM: 48,935 MiB
Power limit: 300 W
Max graphics clock: 3090 MHz
Max memory clock: 14001 MHz
PCIe max: Gen5 x16
```

Disk/storage state during benchmark:

```text
Filesystem size: 160 GB
Used: 67 GB
Available: 94 GB
SoulX-FlashTalk workspace: 62 GB
models/: 53 GB
.venv/: 8.9 GB
```

## Relevant Code Path

Current runs require `--cpu_offload` on this GPU. In `flash_talk/src/pipeline/flash_talk_pipeline.py`, CPU offload means the diffusion model is loaded on CPU and moved to GPU for each generated chunk:

```python
self.model = WanModel.from_pretrained(
    checkpoint_dir,
    device_map='cpu' if self.cpu_offload else self.device,
    torch_dtype=self.param_dtype,
)
```

Then during generation:

```python
if self.cpu_offload:
    self.model.to(self.device)

noise_pred_cond = self.model(...)

if self.cpu_offload:
    self.model.cpu()
    torch.cuda.empty_cache()
    self.vae.model.to(self.device)
```

This means the benchmark includes CPU/GPU transfer overhead, cache clearing, and disabled compile paths. `torch.compile` is only enabled when CPU offload is disabled:

```python
if COMPILE_MODEL and not self.cpu_offload:
    self.model = torch.compile(self.model)
```

## VRAM

Measured with `nvidia-smi` sampled once per second during a short CPU-offload probe run.

```text
Idle VRAM: ~2 MiB
Peak VRAM with --cpu_offload: 39,446 MiB
Peak VRAM with --cpu_offload: ~38.5 GiB
Free VRAM at peak: 8,958 MiB
```

No-offload test failed during model loading:

```text
torch.OutOfMemoryError: CUDA out of memory.
GPU total capacity: 47.27 GiB
Process memory in use at failure: 47.16 GiB
Free memory at failure: ~103 MiB
```

Conclusion:

```text
This 48 GB GPU requires --cpu_offload.
Without CPU offload, SoulX-FlashTalk needs more VRAM than this GPU provides.
```

## Runtime Results

All successful benchmark runs used:

```text
Resolution: 448x768
FPS: 25
sample_steps: 4
sample_shift: 5
audio_encode_mode: stream
--cpu_offload enabled
Blackwell SDPA fallback enabled
```

| Output | Duration | Runtime | Speed |
| --- | ---: | ---: | ---: |
| `github_outputs/voice_tests/res_female_cpu_offload.mp4` | 11.13 sec | ~6 min 37 sec | ~35.7x realtime |
| `github_outputs/voice_tests/res_female_motion_scale_065.mp4` | 11.13 sec | ~6 min 35 sec | ~35.5x realtime |
| `github_outputs/voice_tests/res_male_cpu_offload.mp4` | 11.12 sec | ~6 min 42 sec | ~36.2x realtime |
| `analysis_vram/res_vram_cpu_offload_probe.mp4` | 2.00 sec | ~1 min 48 sec total probe | ~54x realtime including startup/load |

Per generated chunk, denoising usually took:

```text
~36-39 sec per chunk
```

Each chunk runs 4 denoise steps. Individual denoise steps were typically:

```text
~3.35-3.47 sec per denoise step
```

## Cost Baseline

At the current quoted rate:

```text
RTX PRO 5000 Blackwell cost: $7/hour
Measured speed: ~35.5x realtime
```

For 450 minutes of final video:

```text
450 minutes * 35.5 = 15,975 minutes generation
15,975 minutes = 266.25 hours
266.25 hours * $7/hour = $1,863.75
```

Practical budget:

```text
~$1,864 before retries/overhead
~$1,900-$2,100 safer planning range
```

## GB10 Comparison Template

When testing the GB10, run the same command with the same audio, image, prompt, seed, and config.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python generate_video.py \
  --ckpt_dir models/SoulX-FlashTalk-14B \
  --wav2vec_dir models/chinese-wav2vec2-base \
  --input_prompt "A person is talking. Only the foreground character is moving, the background remains static." \
  --cond_image assets/varied_background_selfie_images/01_arab_woman_city_balcony.png \
  --audio_path assets/lumatalk_welcome_female.wav \
  --audio_encode_mode stream \
  --cpu_offload \
  --base_seed 9999 \
  --save_file github_outputs/benchmarks/res_female_gb10_cpu_offload.mp4
```

Record:

```text
GPU name:
VRAM total:
Driver:
Power limit:
Peak VRAM:
Output duration:
Wall-clock runtime:
Realtime factor:
Cost per hour:
Cost per 450 minutes:
Did no-offload work?
```

## Interpretation

For this repo, VRAM is the first gating factor. This RTX PRO 5000 Blackwell has high compute, but because 48 GB VRAM is not enough for no-offload mode, runs are slowed by CPU offload.

A cheaper GPU with more VRAM may be faster overall if it can keep the full model resident on GPU and avoid:

- model CPU/GPU shuttling per chunk
- `torch.cuda.empty_cache()` stalls
- disabled `torch.compile`

Once a GPU has enough VRAM to avoid CPU offload, TFLOPS and memory bandwidth become the main speed factors.
