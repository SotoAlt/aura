# AURA — Audio-Conditioned World Model

## Project Overview

AURA is an audio-conditioned world model that generates and simulates alien corridor environments in real time, driven by audio features. The neural network IS the engine — it learns physics, spatial structure, and dynamics, then predicts future states frame by frame.

- **Studio**: WAWE Studio (Chile)
- **Repo**: github.com/SotoAlt/aura (private)
- **Domain**: waweapps.win
- **Deadline**: April 14, 2026 (demo) / April 30, 2026 (competition)
- **Status**: P1 (Training) — in progress

## Architecture

```
Audio Input → FFT/Feature Extraction → 16-float context vector (c_t)
                                              ↓
          Previous 4 frames (x_{t-4}…x_{t-1}) + c_t + action a_t
                                              ↓
                  DIAMOND U-Net: denoise x_t over K diffusion steps
                                              ↓
                          Denoised frame x_t (64×64 RGB)
                             ↓                ↓
                    feedback (next window)   Three.js → Browser
```

Three systems:
1. **World Model** (Python/PyTorch): DIAMOND v0.3 diffusion world model — pixel-space autoregressive frame prediction with audio conditioning
2. **Browser Client** (Three.js/Vite): Renders decoded frames, captures audio via Web Audio API
3. **Inference Server** (FastAPI): Bridges world model to browser via WebSocket

## Tech Stack

| Layer | Technology |
|-------|-----------|
| World Model | DIAMOND diffusion U-Net (PyTorch) — pixel-space AR frame prediction |
| Training | Google Colab (A100/T4) |
| Data Generation | Custom Python corridor sim (NumPy) |
| Audio Analysis | librosa (training), Web Audio API (inference) |
| 3D Rendering | Three.js |
| Inference Server | FastAPI + WebSocket |
| Client Bundler | Vite |

## Directory Structure

```
aura/
  world_model/                  Python: world model + training pipeline
    diamond/                    ★ DIAMOND v0.3 diffusion world model (PyTorch)
      unet.py                   ★ U-Net with audio-context conditioning
      diffusion.py              ★ DDPM forward/reverse process
      dataset.py                ★ Frame-sequence dataset loader (NPZ)
      train.py                  ★ Training loop (GPU, wandb)
      sample.py                 ★ Autoregressive rollout / dream generation
      eval.py                   Evaluation metrics (FVD, LPIPS, audio correlation)
      utils.py                  Helpers (checkpoint I/O, normalization)
      configs.yaml              DIAMOND training configs (aura + aura_debug)
    dreamer/                    (legacy) Plain JAX cRSSM implementation
      nets.py                   CNN encoder/decoder, MLP, GRU primitives
      rssm.py                   Core: cRSSM with audio context conditioning
      agent.py                  WorldModel + Trainer (optax)
      configs.yaml              AURA-specific training configs (aura + aura_debug)
      checkpoint.py             Save/load model checkpoints (pickle)
      logging.py                Optional wandb logging helpers
    envs/
      corridor.py               ★ Gymnasium env: procedural alien corridor (raycaster)
    audio/
      features.py               ★ librosa: FFT bands, onset, BPM, spectral centroid → 16 floats
      synthetic.py              Procedural audio generation for training
    data/
      generate.py               ★ Episode generator + NPZ dataset loader
    test_pipeline.py            End-to-end P0 verification
    train.py                    Training script (legacy cRSSM)
    eval.py                     Evaluation (legacy cRSSM)
    infer.py                    ★ FastAPI inference server (P2)
    requirements.txt            Local deps (CPU)
    requirements-colab.txt      Colab deps (GPU)
  scripts/
    generate_data.sh            Data generation wrapper (episodes + audio)
    train_diamond.sh            DIAMOND training launcher (Colab / local)
  client/                       Browser: Three.js + Vite
    src/
      audio.js                  Web Audio API, FFT, beat detection
      world.js                  Three.js scene rendering
      net.js                    WebSocket client
      main.js                   Entry point
  server/                       Deferred: WebSocket event server (multiplayer)
  notebooks/
    01_data_generation.ipynb    Colab: generate datasets
    02_train_aura.ipynb         Colab: train world model
    03_eval_dreams.ipynb        Colab: visualize imagined trajectories
  data/                         Generated training data (gitignored)
  checkpoints/                  Model weights (gitignored)
```

★ = key files to read first

## Development Commands

```bash
# Python virtual environment
cd world_model
source .venv/bin/activate

# Run P0 pipeline test (verifies everything works end-to-end)
cd /path/to/aura
python -m world_model.test_pipeline

# Generate training data (local, CPU)
python -m world_model.data.generate --episodes 100 --output data/test
# Or via wrapper script:
bash scripts/generate_data.sh --episodes 100 --output data/test

# Run corridor env standalone
python -m world_model.envs.corridor --episodes 10 --output data/test

# --- DIAMOND v0.3 commands ---

# Train DIAMOND (local CPU smoke test)
python -m world_model.diamond.train --config aura_debug --data data/smoke --steps 100 --checkpoint checkpoints/diamond_smoke.pt --no-wandb
# Or via wrapper script:
bash scripts/train_diamond.sh --config aura_debug --data data/smoke --steps 100

# Train DIAMOND (Colab GPU)
python -m world_model.diamond.train --config aura --data /content/drive/MyDrive/aura/data --steps 50000 --checkpoint /content/drive/MyDrive/aura/checkpoints/diamond_v0.1.pt

# Evaluate DIAMOND checkpoint
python -m world_model.diamond.eval --checkpoint checkpoints/diamond_smoke.pt --output eval_output/diamond_smoke/

# Generate dream rollouts (autoregressive sampling)
python -m world_model.diamond.sample --checkpoint checkpoints/diamond_smoke.pt --steps 64 --output eval_output/dreams/

# --- Legacy cRSSM commands ---

# Train cRSSM (local CPU smoke test)
python -m world_model.train --config aura_debug --data data/smoke --steps 100 --checkpoint checkpoints/smoke.ckpt --no-wandb

# Evaluate cRSSM checkpoint
python -m world_model.eval --checkpoint checkpoints/smoke.ckpt --output eval_output/smoke/

# --- Shared ---

# Run inference server (CPU) — P2
python world_model/infer.py --checkpoint checkpoints/diamond_v0.1.pt

# Client dev server
cd client && npx vite

# Test PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Colab Workflow

```python
# In any Colab notebook:
!git clone https://github.com/SotoAlt/aura.git
%cd aura/world_model
!pip install -r requirements-colab.txt

# Mount Google Drive for data/checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Train DIAMOND on Colab GPU
!python -m world_model.diamond.train \
    --config aura \
    --data /content/drive/MyDrive/aura/data \
    --steps 50000 \
    --checkpoint /content/drive/MyDrive/aura/checkpoints/diamond_v0.1.pt

# Generate dream rollouts
!python -m world_model.diamond.sample \
    --checkpoint /content/drive/MyDrive/aura/checkpoints/diamond_v0.1.pt \
    --steps 64 --output eval_output/dreams/
```

## Audio Context Vector (16 floats)

| Index | Feature | Range | World Effect |
|-------|---------|-------|-------------|
| 0-1 | Sub-bass energy (20-80 Hz) | [0,1] | Organic structure growth |
| 2-3 | Mid energy (250 Hz-2 kHz) | [0,1] | Color saturation, light |
| 4-5 | High frequency (4-20 kHz) | [0,1] | Particle/detail complexity |
| 6-7 | Onset detection | [0,1] | Discrete physics events |
| 8-9 | Estimated BPM (normalized) | [0,1] | Camera forward velocity |
| 10-11 | Spectral centroid | [0,1] | World temperature (palette) |
| 12-13 | RMS energy | [0,1] | Overall activity level |
| 14-15 | Reserved | [0,1] | Future features |

## Constraints

- **Hardware**: 16GB RAM MacBook Air M3, ~21GB disk free
- **No local training** — all training on Google Colab
- **No local GPU** — CPU inference only (JAX `--jax.platform cpu`)
- **Disk budget**: ~1.3GB for project (venv + node_modules + test data)
- **Memory**: Data generation must be episode-at-a-time, never hold full dataset in RAM

## Key References

- [DreamerV3](https://github.com/danijar/dreamerv3) — Base architecture (JAX, MIT license)
- [cRSSM / Dreaming of Many Worlds](https://github.com/sai-prasanna/dreaming_of_many_worlds) — Context conditioning pattern
- [AURA PRD](./docs/AURA_PRD.md) — Full product requirements document

## Development Phases

| Phase | Status | Goal |
|-------|--------|------|
| P0: Foundation | **Complete** | Corridor env + audio pipeline + lightweight cRSSM (plain JAX) |
| P1: Training | **In Progress** | DIAMOND v0.3 diffusion world model — train on Colab, validate audio→world correlation |
| P2: Browser Demo | Pending | FastAPI inference + Three.js client |
| P3: Rhythm Mechanic | Deferred | Beat detection, interactive nodes, scoring |
| P4: Multiplayer | Deferred | WebSocket server, multi-client sync |
