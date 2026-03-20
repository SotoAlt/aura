# Training AURA on Video Data

## Why

The raycaster corridor generates training data that looks like 1993 Doom — dark purple mush. The model can only dream what it's been shown. This workflow extracts frames from a real video (MATSYA / WAWE, 4:39) and applies audio-driven augmentations so the model learns to dream organic visuals that respond to music.

## The Problem

The video is **not** audio-reactive — visuals don't change with the music. If we just pair frames with audio features as-is, the model won't learn audio→visual correlation. We solve this with **audio-driven augmentation**: each frame is transformed based on its paired audio context vector.

## Augmentation Map

| Audio Feature | Index | Visual Effect | Formula |
|---------------|-------|---------------|---------|
| RMS energy | 12-13 | Brightness | `frame * (0.4 + rms * 0.8)` |
| Spectral centroid | 10-11 | Color temperature | Warm (red+) at high, cool (blue+) at low |
| Sub-bass | 0-1 | Saturation | Boost color saturation with bass energy |
| Onset detection | 6-7 | Contrast | Sharpen contrast on beat onsets |
| High frequency | 4-5 | Detail/grain | Add subtle noise at high frequencies |

## Episode Generation Strategy

1. Extract ~2,790 frames at 10fps from video (center-cropped to 64x64)
2. Extract real audio context vectors using `AudioFeatureExtractor`
3. Create sliding windows: 101-frame episodes, stride 20 → ~135 base windows
4. For each window, create 5 augmentation passes:
   - **real**: actual audio features from the music track
   - **high**: sustained high energy (bright, saturated)
   - **low**: sustained low energy (dim, desaturated)
   - **ramp**: energy ramps from low to high
   - **pulse**: alternating high/low every ~2 seconds

Total: 135 × 5 = **675 episodes** (~690MB including raw video)

## Local Workflow

```bash
# 1. Generate episodes from YouTube video
python -m world_model.data.video \
    --url "https://youtu.be/qEJfEyHNZl0" \
    --output data/matsya

# 2. Smoke test training (100 steps, CPU)
python -m world_model.train \
    --config aura_debug \
    --data data/matsya \
    --steps 100 \
    --checkpoint checkpoints/matsya_smoke.ckpt \
    --no-wandb

# 3. Evaluate
python -m world_model.eval \
    --checkpoint checkpoints/matsya_smoke.ckpt \
    --output eval_output/matsya_smoke/
```

## Colab Workflow (GPU Training)

```python
# Step 1: Setup
!git clone https://github.com/SotoAlt/aura.git /content/aura 2>/dev/null || (cd /content/aura && git pull)
%cd /content/aura/world_model
!pip install -q -r requirements-colab.txt
!pip install -q yt-dlp
import jax; print("JAX devices:", jax.devices())

# Step 2: Download video & build episodes (~2 min)
!cd /content/aura && python -m world_model.data.video \
    --url "https://youtu.be/qEJfEyHNZl0" \
    --output /content/video_data

# Step 3: Train (10K steps on T4, ~2 hours)
!cd /content/aura && python -m world_model.train \
    --config aura \
    --data /content/video_data \
    --steps 10000 \
    --checkpoint /content/aura/checkpoints/aura-v0.2.ckpt \
    --no-wandb

# Step 4: Evaluate
!cd /content/aura && python -m world_model.eval \
    --checkpoint /content/aura/checkpoints/aura-v0.2.ckpt \
    --output /content/eval_output/

# Step 5: Save to Drive
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/aura_checkpoints
!cp /content/aura/checkpoints/aura-v0.2.ckpt /content/drive/MyDrive/aura_checkpoints/
!cp -r /content/eval_output/ /content/drive/MyDrive/aura_checkpoints/eval_v0.2/
```

## CLI Reference

```
python -m world_model.data.video [OPTIONS]

Options:
  --url       YouTube video URL (required)
  --output    Output directory for episodes (default: data/matsya)
  --fps       Frame extraction rate (default: 10.0)
  --steps     Steps per episode (default: 100)
  --passes    Augmentation passes per window (default: 5)
  --size      Frame size, square (default: 64)
```

## Verification

After training, the eval script checks 3 success criteria:
1. **no_nan_or_inf** — model outputs are numerically stable
2. **brightness_rms_corr > 0.3** — brightness correlates with RMS energy
3. **high_brighter_than_low** — high-energy dreams are visibly brighter than low-energy

GIFs in the eval output should show organic visuals from the video (not raycaster mush), with visible brightness/color differences between energy levels.

## Files

| File | Role |
|------|------|
| `world_model/data/video.py` | Download, extract, augment, build episodes |
| `world_model/audio/features.py` | `AudioFeatureExtractor` + `unpack_context()` |
| `world_model/data/generate.py` | `NPZDataset` loader (reused as-is) |
| `world_model/train.py` | Training script (works with video data unchanged) |
| `world_model/eval.py` | Evaluation + GIF generation |
