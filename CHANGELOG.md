# Changelog

All notable changes to AURA are documented here.

## [Unreleased] — P1: Training & Validation

### Added
- Video data pipeline (`world_model/data/video.py`) — download YouTube video, extract frames + audio, apply audio-driven augmentations, build NPZ episodes
- Audio-driven augmentation: brightness (RMS), color temperature (centroid), saturation (bass), contrast (onset), grain (high freq)
- Synthetic context profiles (high/low/ramp/pulse) for augmentation variety
- yt-dlp dependency for video download
- Checkpoint save/load (`world_model/dreamer/checkpoint.py`)
- wandb logging helpers (`world_model/dreamer/logging.py`)
- Training script (`world_model/train.py`) — works locally (CPU) and on Colab (GPU)
- Evaluation script (`world_model/eval.py`) — GIF generation + audio correlation metrics
- Shared `preprocess_batch()` in `agent.py` — used by both training and validation
- Colab notebooks for data generation, training, and evaluation
- pyyaml, matplotlib added to Colab requirements

## [0.0.1] — 2026-03-19 — P0: Foundation

### Added
- Conditioned RSSM (cRSSM) world model in plain JAX + optax
- 16-float audio context vector pipeline (librosa)
- Procedural alien corridor environment (raycaster, Gymnasium API)
- Synthetic audio generation for training data
- Episode data generation + NPZ dataset loader
- CNN encoder/decoder, MLP, GRU primitives
- End-to-end P0 verification script (`test_pipeline.py`)
- `aura` (GPU) and `aura_debug` (CPU) training configs
