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
