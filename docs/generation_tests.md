# Generation Test Notes

Date: 2026-07-03

## Environment

- Repo: `/workspace/SoulX-FlashTalk`
- Python environment: `.venv`
- GPU: NVIDIA RTX PRO 5000 Blackwell
- PyTorch: `2.7.1+cu128`
- CUDA runtime used by PyTorch: `12.8`
- FFmpeg: `6.1.1`

The Blackwell GPU hit an `xformers`/FlashAttention fused kernel error during the first generation attempt:

```text
CUDA error ... flash_fwd_launch_template.h:188: invalid argument
```

To complete generation, the repo was patched to use PyTorch scaled dot product attention on compute capability 12.x GPUs.

Changed files:

- `flash_talk/wan/modules/attention.py`
- `flash_talk/infinite_talk/modules/multitalk_attention.py`

## Assets

The selfie archive was unpacked from:

```text
assets/varied_background_selfie_images.zip
```

Into:

```text
assets/varied_background_selfie_images/
```

The generated selfie tests used the same audio file:

```text
sample_results/selfie_10s.wav
```

This file was created from the bundled sample audio:

```text
examples/cantonese_16k.wav
```

## Test Outputs

All videos were generated with:

- Duration: 10.0 seconds
- Resolution: 448x768
- Video codec: H.264
- Audio codec: AAC
- Audio input: `sample_results/selfie_10s.wav`
- Mode: `--audio_encode_mode stream`
- CPU offload: enabled

| Output | Conditioning Image | Size | Approx Runtime |
| --- | --- | ---: | ---: |
| `sample_results/res_selfie_10s.mp4` | `assets/varied_background_selfie_images/01_arab_woman_city_balcony.png` | 848K | 5 min 58 sec |
| `sample_results/res_selfie_04_10s.mp4` | `assets/varied_background_selfie_images/04_filipina_woman_tropical_resort.png` | 842K | 6 min 4 sec |
| `sample_results/res_selfie_08_10s.mp4` | `assets/varied_background_selfie_images/08_south_korean_man_modern_workspace.png` | 802K | 5 min 58 sec |

Each 10-second video generated 9 chunks. Most chunks took about 36-38 seconds.

These selfie test MP4s were also mirrored into a Git-visible folder:

```text
github_outputs/selfie_tests/res_selfie_10s.mp4
github_outputs/selfie_tests/res_selfie_04_10s.mp4
github_outputs/selfie_tests/res_selfie_08_10s.mp4
github_outputs/selfie_tests/res_smoke.mp4
```

## Voice Sample Tests

Two additional tests were run with the provided LumaTalk voice samples. These outputs were saved under `github_outputs/` because `assets/` and `sample_results/` are ignored by Git in this repo.

Input voice files:

| Voice | Source Audio | Duration | Format |
| --- | --- | ---: | --- |
| Female | `assets/lumatalk_welcome_female.wav` | 11.13 sec | WAV, 24 kHz, mono |
| Male | `assets/lumatalk_welcome_male.wav` | 11.12 sec | WAV, 24 kHz, mono |

Generated outputs:

| Output | Conditioning Image | CPU Offload | Duration | Size | Approx Runtime |
| --- | --- | --- | ---: | ---: | ---: |
| `github_outputs/voice_tests/res_female_cpu_offload.mp4` | `assets/varied_background_selfie_images/01_arab_woman_city_balcony.png` | Yes | 11.13 sec | 967K | 6 min 37 sec |
| `github_outputs/voice_tests/res_male_cpu_offload.mp4` | `assets/varied_background_selfie_images/08_south_korean_man_modern_workspace.png` | Yes | 11.12 sec | 904K | 6 min 42 sec |

Validation:

```text
github_outputs/voice_tests/res_female_cpu_offload.mp4
h264 video 448x768
aac audio 24000 Hz mono
duration 11.130000

github_outputs/voice_tests/res_male_cpu_offload.mp4
h264 video 448x768
aac audio 24000 Hz mono
duration 11.120000
```

### Subtle Prompt and Once Audio Encoding Test

A controlled female voice comparison was run without changing the audio. The test used the same female voice sample and same selfie image as `res_female_cpu_offload.mp4`, but changed:

- `--audio_encode_mode once`
- prompt:

```text
A realistic close-up video of a person speaking calmly with subtle natural lip movement. Only small facial motions and gentle head movement occur. The background remains static.
```

Output:

| Output | Audio Encoding | Prompt Style | Duration | Size | Approx Runtime |
| --- | --- | --- | ---: | ---: | ---: |
| `github_outputs/voice_tests/res_female_once_subtle_prompt.mp4` | `once` | Subtle/calm lip motion | 10.28 sec | 886K | 5 min 58 sec |

Validation:

```text
github_outputs/voice_tests/res_female_once_subtle_prompt.mp4
h264 video 448x768
aac audio 24000 Hz mono
duration 10.280000
```

Note: the input female audio is 11.13 seconds, but this `once` mode output muxed to 10.28 seconds because fewer generated video frames were produced than with `stream` mode.

### Audio Motion Scale Test

To reduce overactive lips/teeth without changing the source audio, an inference-time `audio_motion_scale` knob was added.

Implementation:

- `flash_talk/configs/infer_params.yaml`
- `flash_talk/inference.py`
- `flash_talk/src/pipeline/flash_talk_pipeline.py`
- `flash_talk/infinite_talk/modules/multitalk_model.py`

The audio cross-attention residual is now scaled inside each attention block:

```python
x = x + x_a * getattr(self, 'audio_motion_scale', 1.0)
```

Current test value:

```yaml
audio_motion_scale: 0.65
```

Controlled test:

- Same input image as `res_female_cpu_offload.mp4`
- Same female voice sample
- Same original prompt
- Same `stream` audio mode
- Same seed
- Only changed `audio_motion_scale` from implicit `1.0` to `0.65`

Output:

| Output | Audio Motion Scale | Duration | Size | Approx Runtime |
| --- | ---: | ---: | ---: | ---: |
| `github_outputs/voice_tests/res_female_motion_scale_065.mp4` | 0.65 | 11.13 sec | 1.1M | 6 min 35 sec |

Validation:

```text
github_outputs/voice_tests/res_female_motion_scale_065.mp4
h264 video 448x768
aac audio 24000 Hz mono
duration 11.130000
```

Visual note: `audio_motion_scale: 0.65` did not dramatically reduce the over-visible teeth or large mouth shapes. It may be too mild, or the model may be encoding mouth openness through multiple pathways beyond the simple audio residual scale.

## VRAM and Runtime Analysis

A monitored CPU-offload probe was run with a short 2-second audio sample while sampling `nvidia-smi` once per second.

Machine GPU:

```text
NVIDIA RTX PRO 5000 Blackwell
Total VRAM: 48,935 MiB
Idle VRAM: ~2 MiB
```

Observed VRAM during `--cpu_offload` generation:

```text
Peak VRAM used: 39,446 MiB
Peak VRAM used: ~38.5 GiB
Free VRAM at peak: 8,958 MiB
```

Practical conclusion:

```text
With --cpu_offload: budget about 39-40 GB VRAM.
Without --cpu_offload: this 48 GB GPU is not enough.
```

The no-offload test failed during checkpoint loading, before generation:

```text
torch.OutOfMemoryError: CUDA out of memory.
GPU 0 total capacity: 47.27 GiB
Process memory in use at failure: 47.16 GiB
Free memory at failure: ~103 MiB
```

### Long-Run Estimate

The `audio_motion_scale: 0.65` test produced an 11.13-second video in about 6 minutes 35 seconds.

That is approximately:

```text
395 sec generation / 11.13 sec video = ~35.5x realtime
```

At the current settings:

```text
audio_motion_scale: 0.65
sample_steps: 4
audio_encode_mode: stream
--cpu_offload enabled
448x768 output
Blackwell SDPA fallback enabled
```

Estimated wall time for 450 minutes of finished video:

```text
450 minutes video * 35.5 = 15,975 minutes generation
15,975 minutes = 266.25 hours
266.25 hours = ~11.1 days
```

Practical estimate:

```text
450 minutes of output would take roughly 11 days on this machine.
```

This assumes continuous single-GPU generation, no failures, no batching overhead, and the same settings used for `github_outputs/voice_tests/res_female_motion_scale_065.mp4`.

### No CPU Offload Test

A no-offload test was attempted with the female voice sample:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python generate_video.py \
  --ckpt_dir models/SoulX-FlashTalk-14B \
  --wav2vec_dir models/chinese-wav2vec2-base \
  --input_prompt "A person is talking. Only the foreground character is moving, the background remains static." \
  --cond_image assets/varied_background_selfie_images/01_arab_woman_city_balcony.png \
  --audio_path assets/lumatalk_welcome_female.wav \
  --audio_encode_mode stream \
  --save_file github_outputs/voice_tests/res_female_no_cpu_offload.mp4
```

Result: failed during model loading with CUDA out-of-memory before generation started.

Relevant error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 136.00 MiB.
GPU 0 has a total capacity of 47.27 GiB of which 103.81 MiB is free.
Including non-PyTorch memory, this process has 47.16 GiB memory in use.
```

Conclusion: this 48 GB GPU still requires `--cpu_offload` for this model.

## Command Template

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python generate_video.py \
  --ckpt_dir models/SoulX-FlashTalk-14B \
  --wav2vec_dir models/chinese-wav2vec2-base \
  --input_prompt "A person is talking. Only the foreground character is moving, the background remains static." \
  --cond_image assets/varied_background_selfie_images/01_arab_woman_city_balcony.png \
  --audio_path sample_results/selfie_10s.wav \
  --audio_encode_mode stream \
  --cpu_offload \
  --save_file sample_results/res_selfie_10s.mp4
```

Swap `--cond_image` and `--save_file` to test other selfies.

## Validation

The generated files were validated with `ffprobe`:

```text
sample_results/res_selfie_10s.mp4
h264 video 448x768
aac audio
duration 10.000000

sample_results/res_selfie_04_10s.mp4
h264 video 448x768
aac audio
duration 10.000000

sample_results/res_selfie_08_10s.mp4
h264 video 448x768
aac audio
duration 10.000000
```

## TTS

This repo does not include text-to-speech generation. It is audio-driven: an existing audio file is passed through wav2vec and used to drive the generated talking-head motion.
