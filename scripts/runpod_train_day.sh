#!/bin/bash
# AURA Training Day — April 1, 2026
# Runs 4 jobs in parallel on 1 RTX 4090 pod (~$1.80 for 3hrs)
#
# Jobs:
#   A) DIAMOND v2 (64×64, paper-matched, ~1.5hrs)
#   B) JEPA Pong from scratch (ViT-Tiny, ~2.5hrs)
#   C) JEPA Pong fine-tuned from PushT (ViT-Tiny, ~2hrs)
#   D) Quadruped spider data + training (~3hrs)
set -e

echo "========================================="
echo "AURA Training Day — $(date)"
echo "========================================="

# --- Setup ---
echo "[Setup] Installing dependencies..."
pip install -q stable-worldmodel stable-pretraining mujoco h5py hdf5plugin 2>/dev/null
apt-get install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev 2>/dev/null || true
export MUJOCO_GL=egl
export STABLEWM_HOME=/workspace/swm_data

# Clone repo (or pull latest)
cd /workspace
if [ -d aura ]; then
    cd aura && git pull 2>/dev/null || true
else
    git clone https://github.com/SotoAlt/aura.git && cd aura
fi

# Also get stable-worldmodel source (for LeWM training scripts)
cd /workspace
if [ ! -d swm-repo ]; then
    git clone https://github.com/galilai-group/stable-worldmodel.git swm-repo
fi
pip install -e swm-repo/ -q 2>/dev/null

export PYTHONPATH=/workspace/aura:/workspace/swm-repo

echo "[Setup] Done. Starting parallel training jobs..."
echo ""

# =============================================================
# JOB A: DIAMOND v2 (64×64, paper-matched)
# =============================================================
echo "[A] Starting DIAMOND v2..."
(
    cd /workspace/aura
    bash scripts/train_diamond_v2.sh
    echo "[A] DIAMOND v2 COMPLETE"
) > /workspace/log_diamond.txt 2>&1 &
PID_DIAMOND=$!
echo "  DIAMOND PID: $PID_DIAMOND"

# =============================================================
# JOB B+C: JEPA Pong (collect data first, then train 2 models)
# =============================================================
echo "[B/C] Collecting Pong HDF5 data (1000 episodes)..."
cd /workspace/aura
python -u scripts/collect_pong_data.py \
    --episodes 1000 --steps 100 --frameskip 5 \
    --image-size 224 --output pong_train \
    2>&1 | tee /workspace/log_pong_data.txt

echo "[B] Starting JEPA Pong from-scratch..."
(
    cd /workspace/swm-repo
    python scripts/train/lewm.py \
        data.dataset.name=pong_train \
        data.dataset.frameskip=1 \
        data.dataset.num_steps=16 \
        "data.dataset.keys_to_load=[pixels,action]" \
        "data.dataset.keys_to_cache=[action]" \
        trainer.max_epochs=100 \
        trainer.precision=bf16 \
        loader.batch_size=64 \
        wandb.enabled=False \
        checkpointing.dirpath=/workspace/checkpoints/jepa_pong_scratch \
        2>&1
    echo "[B] JEPA Pong from-scratch COMPLETE"
) > /workspace/log_jepa_scratch.txt 2>&1 &
PID_JEPA_SCRATCH=$!
echo "  JEPA scratch PID: $PID_JEPA_SCRATCH"

echo "[C] Starting JEPA Pong fine-tune from PushT..."
(
    cd /workspace/swm-repo
    # Download PushT pretrained weights
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('quentinll/lewm-pusht', 'weights.pt')"

    python scripts/train/lewm.py \
        data.dataset.name=pong_train \
        data.dataset.frameskip=1 \
        data.dataset.num_steps=16 \
        "data.dataset.keys_to_load=[pixels,action]" \
        "data.dataset.keys_to_cache=[action]" \
        trainer.max_epochs=100 \
        trainer.precision=bf16 \
        loader.batch_size=64 \
        wandb.enabled=False \
        checkpointing.dirpath=/workspace/checkpoints/jepa_pong_finetune \
        2>&1
    echo "[C] JEPA Pong fine-tune COMPLETE"
) > /workspace/log_jepa_finetune.txt 2>&1 &
PID_JEPA_FT=$!
echo "  JEPA fine-tune PID: $PID_JEPA_FT"

# =============================================================
# JOB D: Quadruped spider
# =============================================================
echo "[D] Starting Quadruped data collection + training..."
(
    cd /workspace/aura
    # Collect Quadruped data with random policy
    python -u -c "
import stable_worldmodel as swm
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['STABLEWM_HOME'] = '/workspace/swm_data'

print('Creating Quadruped env...')
world = swm.World('swm/QuadrupedDMControl-v0', num_envs=4, image_shape=(224, 224), max_episode_steps=200)

class RandomPolicy:
    def __call__(self, obs):
        return np.random.uniform(-1, 1, (4, 12)).astype(np.float32)

world.set_policy(RandomPolicy())
print('Collecting 500 episodes...')
world.record_dataset('quadruped_random', episodes=500)
world.close()
print('Quadruped data DONE')
" 2>&1

    # Train LeWM on Quadruped
    cd /workspace/swm-repo
    python scripts/train/lewm.py \
        data.dataset.name=quadruped_random \
        data.dataset.frameskip=1 \
        data.dataset.num_steps=16 \
        "data.dataset.keys_to_load=[pixels,action]" \
        "data.dataset.keys_to_cache=[action]" \
        trainer.max_epochs=50 \
        trainer.precision=bf16 \
        loader.batch_size=32 \
        wandb.enabled=False \
        checkpointing.dirpath=/workspace/checkpoints/jepa_quadruped \
        2>&1
    echo "[D] Quadruped COMPLETE"
) > /workspace/log_quadruped.txt 2>&1 &
PID_QUAD=$!
echo "  Quadruped PID: $PID_QUAD"

echo ""
echo "========================================="
echo "All 4 jobs launched. Monitoring..."
echo "========================================="
echo ""

# Monitor loop
while true; do
    sleep 300  # check every 5 min
    echo "--- Status $(date) ---"

    for name_pid in "DIAMOND:$PID_DIAMOND" "JEPA_scratch:$PID_JEPA_SCRATCH" "JEPA_finetune:$PID_JEPA_FT" "Quadruped:$PID_QUAD"; do
        name="${name_pid%%:*}"
        pid="${name_pid##*:}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "  $name: RUNNING"
        else
            echo "  $name: DONE"
        fi
    done

    # Check if all done
    all_done=true
    for pid in $PID_DIAMOND $PID_JEPA_SCRATCH $PID_JEPA_FT $PID_QUAD; do
        if kill -0 "$pid" 2>/dev/null; then
            all_done=false
        fi
    done
    if $all_done; then
        echo ""
        echo "========================================="
        echo "ALL TRAINING COMPLETE — $(date)"
        echo "========================================="
        break
    fi
done

# List outputs
echo ""
echo "Checkpoints:"
ls -lh /workspace/aura/checkpoints/ 2>/dev/null
ls -lh /workspace/checkpoints/*/ 2>/dev/null
echo ""
echo "Logs:"
for log in /workspace/log_*.txt; do
    echo "--- $log (last 3 lines) ---"
    tail -3 "$log"
done
