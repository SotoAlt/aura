#!/bin/bash
set -e
echo "=== DIAMOND Training on RunPod ==="

cd /workspace/aura && git pull

# Generate bright training data
echo "Generating bright corridor data..."
PYTHONPATH=/workspace/aura python -m world_model.envs.abstract_visual \
    --episodes 200 --steps 200 --output /workspace/diamond_bright --size 128

# Train DIAMOND
echo "Training DIAMOND 128x128..."
PYTHONPATH=/workspace/aura python -m world_model.diamond.train \
    --config aura_abstract_128 \
    --data /workspace/diamond_bright \
    --steps 30000 \
    --checkpoint /workspace/aura/checkpoints/diamond_bright_v1.ckpt \
    --device auto \
    --log-every 500 \
    --save-every 5000

echo "Done! Checkpoint at /workspace/aura/checkpoints/diamond_bright_v1.ckpt"
