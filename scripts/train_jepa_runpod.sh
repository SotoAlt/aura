#!/bin/bash
# AURA ASCII JEPA training on RunPod
#
# Usage:
#   1. Deploy a RunPod pod (RTX 4090 or any GPU)
#   2. SSH in
#   3. Run: bash /workspace/aura/scripts/train_jepa_runpod.sh
#
# Cost: RTX 4090 ($0.39/hr) x ~20 min = ~$0.13

set -e

EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
CKPT="/workspace/aura/checkpoints/ascii_jepa_v2.pt"

echo "============================================"
echo "  AURA ASCII JEPA Training (RunPod)"
echo "============================================"
echo "Epochs:     ${EPOCHS}"
echo "Batch:      ${BATCH}"
echo "Checkpoint: ${CKPT}"
echo ""

# Step 0: Clone or update repo
if [ ! -d /workspace/aura ]; then
    echo "=== Cloning repo ==="
    git clone https://github.com/SotoAlt/aura.git /workspace/aura
else
    echo "=== Updating repo ==="
    cd /workspace/aura && git pull
fi
cd /workspace/aura

# Step 1: Install deps (torch should be pre-installed on RunPod)
echo "=== Installing deps ==="
pip install -q numpy PyYAML tqdm 2>/dev/null

# Step 2: Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = getattr(torch.cuda.get_device_properties(0), 'total_memory', 0)
    print(f'Memory: {mem / 1e9:.1f} GB')
"

# Step 3: Check if data exists, generate if not
if [ ! -f data/ascii_training.npz ]; then
    echo "=== Generating ASCII training data ==="
    # Generate v2 data if the script exists
    if [ -f scripts/generate_ascii_data_v2.sh ]; then
        bash scripts/generate_ascii_data_v2.sh
        DATA="data/ascii_training_v2.npz"
    else
        python -m world_model.envs.ascii_corridor generate \
            --episodes 100 --steps 200 \
            --output data/ascii_training.jsonl
        # Convert to NPZ
        python -c "
import json, numpy as np, torch
from world_model.ascii_model.model import frame_to_indices
frames, audios, episodes = [], [], []
with open('data/ascii_training.jsonl') as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        idx = frame_to_indices(r['ascii_frame'])
        if isinstance(idx, torch.Tensor): idx = idx.numpy()
        frames.append(idx)
        audios.append(r['audio_context'])
        episodes.append(r.get('episode', 0))
np.savez_compressed('data/ascii_training.npz',
    frames=np.array(frames, dtype=np.int16),
    audios=np.array(audios, dtype=np.float32),
    episodes=np.array(episodes, dtype=np.int32))
print(f'Generated {len(frames)} frames')
"
        DATA="data/ascii_training.npz"
    fi
else
    DATA="data/ascii_training.npz"
    echo "=== Using existing data: ${DATA} ==="
fi

# Use v2 data if available
[ -f data/ascii_training_v2.npz ] && DATA="data/ascii_training_v2.npz"
echo "Training data: ${DATA}"

# Step 4: Train
echo ""
echo "=== Training JEPA (${EPOCHS} epochs) ==="
mkdir -p checkpoints

# Resume if checkpoint exists
RESUME_MSG="Starting fresh"
[ -f "${CKPT}" ] && RESUME_MSG="Resuming from existing checkpoint"
echo "${RESUME_MSG}"

python -m world_model.ascii_model.jepa_model \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --checkpoint "${CKPT}" \
    --device cuda

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: ${CKPT}"
echo ""
echo "  To download checkpoint:"
echo "  scp -P <port> root@<ip>:${CKPT} ."
echo "============================================"
