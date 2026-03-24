#!/bin/bash
# Train DIAMOND 128x128 on abstract sci-fi corridor data + JEPA v3
# Run on RunPod: git clone + bash this script
set -e

STEPS="${STEPS:-30000}"
CKPT_DIR="/workspace/aura/checkpoints"

echo "============================================"
echo "  AURA — Abstract 128x128 + JEPA v3"
echo "============================================"

# Clone/update
if [ ! -d /workspace/aura ]; then
    git clone https://github.com/SotoAlt/aura.git /workspace/aura
fi
cd /workspace/aura && git pull

# Install deps
pip install -q Pillow PyYAML tqdm librosa numpy 2>/dev/null

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Step 1: Generate abstract 128x128 training data
echo ""
echo "=== Step 1: Generate 128x128 training data ==="
if [ -d /workspace/abstract_128 ] && [ $(ls /workspace/abstract_128/episode_*.npz 2>/dev/null | wc -l) -gt 50 ]; then
    echo "Data exists, skipping..."
else
    python -m world_model.envs.abstract_visual --episodes 200 --steps 200 --output /workspace/abstract_128 --size 128
fi

EPISODE_COUNT=$(ls -1 /workspace/abstract_128/episode_*.npz 2>/dev/null | wc -l)
echo "Episodes: ${EPISODE_COUNT}"

# Step 2: Train DIAMOND 128x128
echo ""
echo "=== Step 2: Train DIAMOND 128x128 (${STEPS} steps) ==="
mkdir -p ${CKPT_DIR}

DIAMOND_CKPT="${CKPT_DIR}/diamond_abstract_128.ckpt"
RESUME_FLAG=""
if [ -f "${DIAMOND_CKPT}" ]; then
    echo "Resuming from existing checkpoint"
    RESUME_FLAG="--resume ${DIAMOND_CKPT}"
fi

PYTHONPATH=/workspace/aura python -m world_model.diamond.train \
    --config aura_abstract_128 \
    --data /workspace/abstract_128 \
    --steps ${STEPS} \
    --checkpoint ${DIAMOND_CKPT} \
    ${RESUME_FLAG} \
    --device auto \
    --log-every 500 \
    --save-every 5000

# Step 3: Eval DIAMOND
echo ""
echo "=== Step 3: Eval DIAMOND 128x128 ==="
PYTHONPATH=/workspace/aura python -c "
import numpy as np, torch
from pathlib import Path
from PIL import Image
from world_model.diamond.sample import load_model, imagine

model, cfg = load_model('${DIAMOND_CKPT}')
device = next(model.parameters()).device
C = cfg['context_frames']
eps = sorted(Path('/workspace/abstract_128').glob('episode_*.npz'))
ep = np.load(eps[len(eps)//2])
seed = ep['image'][:C]

scenarios = {
    'high_energy': np.tile([[0.9]*2+[0.9]*2+[0.5]*2+[0.0]*2+[0.6]*2+[0.5]*2+[0.9]*2+[0.0]*2], (50,1)).astype(np.float32),
    'low_energy': np.tile([[0.05]*2+[0.05]*2+[0.05]*2+[0.0]*2+[0.3]*2+[0.3]*2+[0.05]*2+[0.0]*2], (50,1)).astype(np.float32),
    'forward_motion': np.tile([[0.4]*2+[0.5]*2+[0.3]*2+[0.0]*2+[0.6]*2+[0.5]*2+[0.6]*2+[0.0]*2], (50,1)).astype(np.float32),
}

Path('/workspace/eval_abstract').mkdir(exist_ok=True)
for name, ctx in scenarios.items():
    print(f'{name}...', end=' ', flush=True)
    frames = imagine(model, seed, ctx, 50, device)
    imgs = [Image.fromarray(f).resize((256,256), Image.NEAREST) for f in frames]
    imgs[0].save(f'/workspace/eval_abstract/{name}.gif', save_all=True, append_images=imgs[1:], duration=100, loop=0)
    ff = frames.astype(np.float32)/255.0
    print(f'brightness={np.mean(ff):.3f} flow={np.mean(np.abs(np.diff(ff,axis=0))):.4f}')
print('GIFs at /workspace/eval_abstract/')
"

# Step 4: Train JEPA v3 with diverse ASCII data
echo ""
echo "=== Step 4: Train JEPA v3 (50 epochs) ==="
PYTHONPATH=/workspace/aura python -m world_model.ascii_model.jepa_model \
    --data data/ascii_training_v2.npz \
    --epochs 50 \
    --batch-size 64 \
    --checkpoint ${CKPT_DIR}/ascii_jepa_v3.pt \
    --device cuda

echo ""
echo "============================================"
echo "  All training complete!"
echo "  DIAMOND 128: ${DIAMOND_CKPT}"
echo "  JEPA v3: ${CKPT_DIR}/ascii_jepa_v3.pt"
echo "  GIFs: /workspace/eval_abstract/"
echo "============================================"
