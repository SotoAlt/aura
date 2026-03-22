#!/bin/bash
# AURA DIAMOND training script for Google Colab
# Usage: bash scripts/train_diamond.sh
#
# Expects:
#   - Running in /content/aura (git cloned)
#   - Video data already at /content/video_data/ (from previous pipeline)
#   - Or set DATA_DIR env var to point to episode data
set -e

DATA_DIR="${DATA_DIR:-/content/video_data}"
STEPS="${AURA_STEPS:-30000}"
CKPT_DIR="/content/aura/checkpoints"
EVAL_DIR="/content/eval_diamond"

echo "============================================"
echo "  AURA DIAMOND Training Pipeline"
echo "============================================"
echo "Data:       ${DATA_DIR}"
echo "Steps:      ${STEPS}"
echo "Checkpoint: ${CKPT_DIR}/diamond.ckpt"
echo ""

# Step 0: Install PyTorch deps
echo "=== Step 0: Install dependencies ==="
pip install -q -r /content/aura/world_model/requirements-diamond.txt

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = getattr(torch.cuda.get_device_properties(0), 'total_memory', 0) or getattr(torch.cuda.get_device_properties(0), 'total_mem', 0)
    print(f'Memory: {mem / 1e9:.1f} GB')
"

# Step 1: Check data exists
echo ""
echo "=== Step 1: Verify training data ==="
EPISODE_COUNT=$(ls -1 "${DATA_DIR}"/episode_*.npz 2>/dev/null | wc -l)
if [ "$EPISODE_COUNT" -eq 0 ]; then
    echo "ERROR: No episodes found in ${DATA_DIR}"
    echo "Run the video data pipeline first, or set DATA_DIR"
    exit 1
fi
echo "Found ${EPISODE_COUNT} episodes"

# Step 2: Train
echo ""
echo "=== Step 2: Train DIAMOND (${STEPS} steps) ==="
mkdir -p "${CKPT_DIR}"

python -m world_model.diamond.train \
    --config aura_diamond \
    --data "${DATA_DIR}" \
    --steps "${STEPS}" \
    --checkpoint "${CKPT_DIR}/diamond.ckpt" \
    --device auto \
    --log-every 100 \
    --save-every 5000

# Step 3: Evaluate
echo ""
echo "=== Step 3: Evaluate ==="
python -m world_model.diamond.eval \
    --checkpoint "${CKPT_DIR}/diamond.ckpt" \
    --data "${DATA_DIR}" \
    --output "${EVAL_DIR}" \
    --frames 50

# Step 4: Package results
echo ""
echo "=== Step 4: Package results ==="
python -c "
import shutil
shutil.make_archive('/content/eval_diamond_results', 'zip', '${EVAL_DIR}')
print('Results zipped to /content/eval_diamond_results.zip')
print()
print('Download with:')
print('  from google.colab import files')
print('  files.download(\"/content/eval_diamond_results.zip\")')
"

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: ${CKPT_DIR}/diamond.ckpt"
echo "  GIFs + metrics: ${EVAL_DIR}/"
echo "============================================"
