#!/bin/bash
# AURA DIAMOND training script for RunPod GPU instances
#
# Usage:
#   1. Create a RunPod pod with RTX 4090 or A100
#   2. SSH in: ssh root@<pod-ip> -p <port>
#   3. Run: bash /workspace/train_runpod.sh
#      Or if repo not yet cloned: wget -qO- https://raw.githubusercontent.com/SotoAlt/aura/main/scripts/train_runpod.sh | bash
#
# Cost estimate:
#   RTX 4090 ($0.39/hr) x 1.5 hrs = ~$0.60
#   A100 ($0.79/hr) x 0.75 hrs = ~$0.60
#
# Environment variables:
#   STEPS       — training steps (default: 30000)
#   VIDEO_URL   — video URL to download (default: https://0x0.st/Pp_q.mp4)
set -e

STEPS="${STEPS:-30000}"
VIDEO_URL="${VIDEO_URL:-https://0x0.st/Pp_q.mp4}"
WORK="/workspace"
REPO="${WORK}/aura"
DATA_DIR="${WORK}/video_data"
CKPT_DIR="${REPO}/checkpoints"
EVAL_DIR="${WORK}/eval_diamond"
CKPT_PATH="${CKPT_DIR}/diamond.ckpt"

echo "============================================"
echo "  AURA DIAMOND — RunPod Training Pipeline"
echo "============================================"
echo "Steps:      ${STEPS}"
echo "Video:      ${VIDEO_URL}"
echo "Checkpoint: ${CKPT_PATH}"
echo ""

# --- Step 0: Install dependencies ---
echo "=== Step 0: Install dependencies ==="
pip install numpy Pillow PyYAML tqdm librosa einops
# torch/torchvision are pre-installed on RunPod images

# --- Step 1: Clone repo ---
echo ""
echo "=== Step 1: Clone repo ==="
if [ -d "${REPO}/.git" ]; then
    echo "Repo already cloned, pulling latest..."
    cd "${REPO}" && git pull && cd "${WORK}"
else
    git clone https://github.com/SotoAlt/aura.git "${REPO}"
fi

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
    print(f'Memory: {mem / 1e9:.1f} GB')
    rate = 0.79 if mem > 40e9 else 0.39
    hours = ${STEPS} / 20000  # rough estimate
    print(f'Estimated cost: \${rate * hours:.2f} ({hours:.1f} hrs @ \${rate}/hr)')
"

# --- Step 2: Download video and extract frames + audio ---
echo ""
echo "=== Step 2: Download video & extract frames ==="
RAW_DIR="${DATA_DIR}/_raw"
mkdir -p "${RAW_DIR}/frames"

VIDEO_PATH="${RAW_DIR}/video.mp4"
AUDIO_PATH="${RAW_DIR}/audio.wav"
FRAMES_DIR="${RAW_DIR}/frames"

if [ ! -f "${VIDEO_PATH}" ]; then
    echo "Downloading video..."
    wget -q --show-progress -O "${VIDEO_PATH}" "${VIDEO_URL}"
else
    echo "Video already downloaded: ${VIDEO_PATH}"
fi

if [ ! -f "${AUDIO_PATH}" ]; then
    echo "Extracting audio (mono, 22050 Hz)..."
    ffmpeg -y -i "${VIDEO_PATH}" -ar 22050 -ac 1 "${AUDIO_PATH}" 2>/dev/null
fi

FRAME_COUNT=$(ls -1 "${FRAMES_DIR}"/frame_*.png 2>/dev/null | wc -l)
if [ "${FRAME_COUNT}" -eq 0 ]; then
    echo "Extracting frames (64x64, 10fps)..."
    ffmpeg -y -i "${VIDEO_PATH}" \
        -vf "crop=min(iw\,ih):min(iw\,ih),scale=64:64:flags=lanczos" \
        -r 10 "${FRAMES_DIR}/frame_%05d.png" 2>/dev/null
    FRAME_COUNT=$(ls -1 "${FRAMES_DIR}"/frame_*.png | wc -l)
fi
echo "Frames ready: ${FRAME_COUNT}"

# --- Step 3: Build episodes ---
echo ""
echo "=== Step 3: Build episodes ==="
EPISODE_COUNT=$(ls -1 "${DATA_DIR}"/episode_*.npz 2>/dev/null | wc -l)
if [ "${EPISODE_COUNT}" -eq 0 ]; then
    cd "${REPO}"
    python -c "
from pathlib import Path
from world_model.data.video import build_episodes
build_episodes(
    frames_dir=Path('${FRAMES_DIR}'),
    audio_path=Path('${AUDIO_PATH}'),
    output_dir='${DATA_DIR}',
    steps_per_episode=100,
    augmentation_passes=5,
    fps=10.0,
)
"
    EPISODE_COUNT=$(ls -1 "${DATA_DIR}"/episode_*.npz | wc -l)
else
    echo "Episodes already built: ${EPISODE_COUNT}"
fi
echo "Total episodes: ${EPISODE_COUNT}"

# --- Step 4: Train DIAMOND ---
echo ""
echo "=== Step 4: Train DIAMOND (${STEPS} steps) ==="
mkdir -p "${CKPT_DIR}"
cd "${REPO}"

RESUME_FLAG=""
if [ -f "${CKPT_PATH}" ]; then
    echo "Resuming from existing checkpoint: ${CKPT_PATH}"
    RESUME_FLAG="--resume"
fi

python -m world_model.diamond.train \
    --config aura_diamond \
    --data "${DATA_DIR}" \
    --steps "${STEPS}" \
    --checkpoint "${CKPT_PATH}" \
    --device auto \
    --log-every 100 \
    --save-every 5000 \
    ${RESUME_FLAG}

# --- Step 5: Evaluate ---
echo ""
echo "=== Step 5: Evaluate ==="
python -m world_model.diamond.eval \
    --checkpoint "${CKPT_PATH}" \
    --data "${DATA_DIR}" \
    --output "${EVAL_DIR}" \
    --frames 50

# --- Step 6: Summary ---
echo ""
echo "============================================"
echo "  Training complete!"
echo "============================================"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  GIFs + metrics: ${EVAL_DIR}/"
echo "  Episodes: ${EPISODE_COUNT}"
echo ""
echo "  To download checkpoint:"
echo "    scp -P <port> root@<pod-ip>:${CKPT_PATH} ."
echo ""
echo "  To resume with more steps:"
echo "    STEPS=50000 bash ${REPO}/scripts/train_runpod.sh"
echo "============================================"
