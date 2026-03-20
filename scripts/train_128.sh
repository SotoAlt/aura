#!/bin/bash
# AURA 128x128 training script for Colab
# Usage: bash scripts/train_128.sh
set -e

echo "=== Step 1: Reuse existing video, re-extract at 128x128 ==="
mkdir -p /content/video_data_128/_raw/frames
cp /content/video_data/_raw/video.mp4 /content/video_data_128/_raw/
cp /content/video_data/_raw/audio.wav /content/video_data_128/_raw/
ffmpeg -y -i /content/video_data_128/_raw/video.mp4 \
  -vf "crop=min(iw\,ih):min(iw\,ih),scale=128:128:flags=lanczos" \
  -r 10 /content/video_data_128/_raw/frames/frame_%05d.png 2>&1 | tail -3

echo "=== Step 2: Build episodes ==="
cd /content/aura
python -c "
from world_model.data.video import load_frames, build_episodes
from pathlib import Path
build_episodes(
    Path('/content/video_data_128/_raw/frames'),
    Path('/content/video_data_128/_raw/audio.wav'),
    '/content/video_data_128',
)
"

echo "=== Step 3: Train 128x128 (5K steps) ==="
python -m world_model.train \
  --config aura_128 \
  --data /content/video_data_128 \
  --steps ${AURA_STEPS:-5000} \
  --checkpoint /content/aura/checkpoints/aura-v0.3-128.ckpt \
  --no-wandb

echo "=== Step 4: Evaluate (seeded from real video frames) ==="
python -m world_model.eval \
  --checkpoint /content/aura/checkpoints/aura-v0.3-128.ckpt \
  --output /content/eval_128/ \
  --seed-data /content/video_data_128

echo "=== Done! Run this in a Python cell to download: ==="
echo "import shutil; shutil.make_archive('/content/eval_128_gifs','zip','/content/eval_128')"
echo "from google.colab import files; files.download('/content/eval_128_gifs.zip')"
