#!/bin/bash
set -e
echo "=== DIAMOND v2 Training (64x64, paper-matched) ==="

cd /workspace/aura

# Generate 64x64 corridor data (500 episodes)
echo "Generating 64x64 corridor data..."
PYTHONPATH=/workspace/aura python -m world_model.envs.abstract_visual \
    --episodes 500 --steps 200 --output /workspace/diamond_64 --size 64

echo "Data generated. Starting training..."

# Train DIAMOND v2 — 64x64, 4.4M params, paper-matched config
PYTHONPATH=/workspace/aura python -u -m world_model.diamond.train \
    --config aura_diamond_v2 \
    --data /workspace/diamond_64 \
    --steps 30000 \
    --checkpoint /workspace/aura/checkpoints/diamond_v2.ckpt \
    --device auto \
    --log-every 500 \
    --save-every 5000

echo "DIAMOND v2 DONE! Checkpoint at /workspace/aura/checkpoints/diamond_v2.ckpt"
