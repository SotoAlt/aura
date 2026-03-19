# AURA — Audio-Conditioned World Model

A neural network that learns to simulate a physically consistent 3D environment where audio features drive world physics in real time.

**WAWE Studio** · 2026

## What is AURA?

AURA is an audio-conditioned world model based on DreamerV3's RSSM architecture, extended with context conditioning (cRSSM). Audio features — bass energy, tempo, spectral density — are injected as continuous physics parameters that modify the world model's learned dynamics. The neural network IS the engine.

## Architecture

- **World Model**: DreamerV3 fork (JAX) with cRSSM audio context conditioning
- **Browser Client**: Three.js + Web Audio API + Vite
- **Inference Server**: FastAPI, serves predicted frames via WebSocket

## Quick Start

```bash
# Python environment (local dev)
cd world_model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Browser client
cd client
npm install
npx vite
```

## Training

Training runs on Google Colab. See `notebooks/` for the full pipeline:
1. `01_data_generation.ipynb` — Generate synthetic corridor + audio datasets
2. `02_train_aura.ipynb` — Train the world model
3. `03_eval_dreams.ipynb` — Visualize imagined trajectories

## Key Papers

- [DreamerV3](https://github.com/danijar/dreamerv3) — Base architecture
- [Dreaming of Many Worlds (cRSSM)](https://github.com/sai-prasanna/dreaming_of_many_worlds) — Context conditioning mechanism
