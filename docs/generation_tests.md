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
