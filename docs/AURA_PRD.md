# AURA — Audio-Conditioned World Model
## Product Requirements Document v0.2
**WAWE Studio** · April 2026 · Internal

---

## Document Info

| Field | Value |
|-------|-------|
| Project | AURA — Audio-conditioned World Model |
| Studio | WAWE Studio (Chile) |
| Status | P1 Training — JEPA validated, DIAMOND experimental |
| Target | Playable browser demos + trained world models |
| Deadline | April 14, 2026 (demo); April 30 (competition) |
| Compute | RunPod RTX 4090 / A40 ($0.22-0.59/hr) |
| Open Source | Training pipeline: yes. Model weights: TBD |

---

## 1. Vision

AURA is an audio-conditioned world model project exploring how neural networks can learn physics, predict dynamics, and plan actions — all from pixels. The project builds increasingly complex demos, each proving a capability:

1. **Pong** — JEPA predicts ball position (validated, LeCun-approved)
2. **Spider** — JEPA plans locomotion for a 4-legged robot
3. **Rail Corridor** — MuJoCo 3D corridor with obstacles, audio-reactive
4. **Rail Shooter** — Full game combining prediction + planning + audio

**What makes AURA different:**
- Audio is not a visual effect — it drives world dynamics (obstacle spawn, speed, complexity)
- The world model predicts in latent space, plans via CEM, renders via physics engine
- Runs on consumer CPU (~15fps JEPA inference on M3 MacBook)
- Building blocks: each demo validates a capability needed for the next

---

## 2. Architecture (Current)

### 2.1 Two World Model Approaches

**JEPA (LeWM) — Primary, Working**
- Joint Embedding Predictive Architecture from "LeWorldModel" paper (arxiv:2603.19312)
- ViT-Tiny encoder (12 layers, 192-dim, patch_size=14) + 6-layer causal Transformer predictor
- ~18M params. Predicts next state in latent space, NOT pixels
- State extracted via trained probe MLP
- CEM planning: sample 300 action sequences → rollout in latent space → pick best
- Runs on CPU at ~15fps
- Ground truth: github.com/lucas-maes/le-wm, github.com/galilai-group/stable-worldmodel, HF quentinll/lewm

**DIAMOND — Experimental, Needs More Training**
- Diffusion U-Net world model (EDM framework, Karras et al.)
- Generates actual pixel frames conditioned on audio features
- Paper-matched config: 64×64, [64,64,64,64] channels, no attention, ~4.4M params
- Needs 100K+ training steps for sharp output (30K produced flat colors)
- Reference: github.com/eloialonso/diamond

### 2.2 System Architecture

```
User Input (mouse/keyboard/audio)
        ↓
Browser Client (HTML/Canvas/WebSocket)
        ↓ WebSocket
Inference Server (FastAPI, Python)
        ↓
Physics Engine (PongWorld / MuJoCo)  ←→  JEPA World Model (prediction + planning)
        ↓                                         ↓
Rendered Frame                           Ghost/prediction overlay
        ↓
Browser displays frame + JEPA prediction
```

### 2.3 Audio Feature Vector (16 floats)

| Index | Feature | Range | World Effect |
|-------|---------|-------|-------------|
| 0-1 | Sub-bass energy (20-80 Hz) | [0,1] | Obstacle density, structure growth |
| 2-3 | Mid energy (250 Hz-2 kHz) | [0,1] | Color saturation, light |
| 4-5 | High frequency (4-20 kHz) | [0,1] | Particle/detail complexity |
| 6-7 | Onset detection | [0,1] | Discrete events (projectile spawn) |
| 8-9 | Estimated BPM (normalized) | [0,1] | Camera/rail forward velocity |
| 10-11 | Spectral centroid | [0,1] | World temperature (palette) |
| 12-13 | RMS energy | [0,1] | Overall activity level |
| 14-15 | Reserved | [0,1] | Future features |

---

## 3. Tech Stack

| Layer | Technology |
|-------|-----------|
| World Model (JEPA) | LeWM ViT-Tiny (PyTorch), stable-worldmodel framework |
| World Model (DIAMOND) | EDM diffusion U-Net (PyTorch) |
| Physics | PongWorld (custom), MuJoCo (corridors/robots) |
| Training | RunPod RTX 4090 / A40, le-wm train.py |
| Data | HDF5 (stable-worldmodel format), 224×224 RGB |
| Audio Analysis | librosa (training), Web Audio API (inference) |
| Inference Server | FastAPI + WebSocket (Python, CPU) |
| Browser Client | HTML5 Canvas, vanilla JS |
| Planning | CEM solver (stable-worldmodel) |

---

## 4. Completed Work

### 4.1 Experiments Run

| Experiment | Result | Lesson |
|-----------|--------|--------|
| Bounce ball JEPA | Ball too simple, JEPA unnecessary | Need complex enough dynamics |
| Golf JEPA | AR collapse after 1 step | Latent chaining doesn't work |
| Pool JEPA (7 balls) | L_pred=0.003 but probe fails (28 vars) | Too many state variables |
| Pong JEPA (CNN, 13M) | ball_x=0.99 corr, ghost ball demo works | Smooth dynamics + few vars = success |
| Pong JEPA (ViT-Tiny frozen PushT) | Predictor cos=0.996 but probe ball_x=0.76 | Frozen encoder features don't transfer for probe |
| Pong JEPA (ViT-Tiny from scratch) | cos≈0 — frameskip bug (action_dim=2 vs 10) | ALWAYS match frameskip in config to data collection |
| DIAMOND v1 (128×128, 30K) | Blurry — sigma_max=80 wrong, model 29M too big | Must match paper config exactly |
| DIAMOND v2 (64×64, 30K) | Better range but flat colors — 30K not enough | Need 100K+ steps for spatial structure |
| LeWM PushT pretrained on Pong frames | 0.988 cosine — ViT-Tiny transfers across envs | Pretrained encoder is universal |

### 4.2 Key Findings (Mistakes NOT Repeated)

1. **NEVER celebrate L_pred alone** — validate full pipeline (encode → predict → probe → render)
2. **Probe trained on encoder outputs ≠ predictor outputs** — domain gap from different projectors
3. **Zero actions at inference when trained with real actions** — out-of-distribution conditioning
4. **Feedback loop (JEPA as physics)** — error compounds when re-encoding own predictions
5. **frameskip in training config MUST match data collection** — wrong frameskip gives wrong action_dim
6. **DIAMOND sigma_max=80 is wrong** — paper uses training=20, inference=5.0
7. **DIAMOND needs [64,64,64,64] channels, NO attention** — our [64,128,256,256] was 7x too big
8. **Clear embedding buffer on ball reset** — stale context after scoring events
9. **Always show accuracy vs baseline** — "85% accuracy" is meaningless without comparison

### 4.3 Honesty Rules

- JEPA predicts in latent space — it is NOT a frame generator
- The probe extracts approximate state from latent predictions — not exact
- CEM planning is what makes JEPA useful for games — not the probe
- DIAMOND generates pixels but needs GPU for real-time
- Our CNN JEPA (13M) works for Pong ghost ball but has weak encoder vs ViT-Tiny
- Claims must match what the code actually does — no fake overlays

---

## 5. Current Training (April 1, 2026)

| Job | Pod | Progress | Notes |
|-----|-----|----------|-------|
| JEPA Pong ViT-Tiny v3 | A40 Canada $0.40/hr | Epoch 0/100, frameskip=5 FIXED | Using le-wm paper code |
| DIAMOND 200K | RTX 3090 US $0.22/hr | Step 10K/200K | Checkpoints every 10K, test at 50K+ |

### 5.1 JEPA Validation Criteria
- L_pred < 0.01
- 1-step cosine > 0.95
- 5-step rollout cosine > 0.8
- Probe: ball_x > 0.95, ball_y > 0.95
- Ghost ball in demo noticeably better than baseline

### 5.2 DIAMOND Validation Criteria
- Output frames show corridor walls/floor geometry (not flat colors)
- Audio conditioning changes output (different RMS → different brightness)
- AR rollout 16 frames maintains consistency
- Pixel range uses full [0, 255]

---

## 6. Roadmap

### Week 1 (March 28 - April 4) — Foundation ✅
- [x] Pong JEPA with CNN encoder (ghost ball demo)
- [x] DIAMOND v1 training (blurry but learned)
- [x] Honesty audit of JEPA pipeline
- [x] DIAMOND v2 with paper-matched config
- [x] stable-worldmodel integration
- [x] LeWM pretrained model testing
- [ ] JEPA Pong with ViT-Tiny (training now)
- [ ] DIAMOND 200K (training overnight)

### Week 2 (April 5 - April 11) — New Environments
- [ ] Quadruped spider: data collection + LeWM training
- [ ] MuJoCo rail corridor v0.1: scene + Gymnasium env + data collection
- [ ] Rail corridor LeWM training
- [ ] Browser demos for spider + corridor
- [ ] CEM planning integration (paper's actual approach)

### Week 3 (April 12 - April 14) — Demo Polish
- [ ] Best JEPA model → polished Pong demo
- [ ] Audio-reactive corridor (DIAMOND or JEPA-based)
- [ ] Spider game prototype
- [ ] Demo video recording
- [ ] Deploy inference server

### April 30 — Competition
- [ ] Rail shooter prototype (corridor + character + projectiles)
- [ ] Audio driving everything (obstacle spawn, speed, complexity)
- [ ] Documentation + paper

---

## 7. Repository Structure (Current)

```
aura/
  world_model/
    ascii_model/
      jepa_pool.py          ★ JEPA architecture (CNN + ViT variants)
    diamond/
      unet.py               ★ DIAMOND U-Net
      diffusion.py          ★ EDM diffusion (training + sampling)
      train.py              Training loop
      configs.yaml          All DIAMOND configs (v1, v2, paper-matched)
    envs/
      pong_world.py         ★ Pong physics engine + renderer
      pong_gym.py           Pong as Gymnasium env (224×224, for LeWM)
      abstract_visual.py    Alien corridor renderer (for DIAMOND)
    audio/
      features.py           Audio feature extraction (librosa)
    infer_pong.py           ★ Pong JEPA inference server (WebSocket)
    infer_diamond.py        DIAMOND inference server (WebSocket)
  client/
    pong/index.html         ★ Pong game with JEPA ghost ball
    diamond/index.html      DIAMOND audio-reactive corridor
  scripts/
    collect_pong_data.py    HDF5 data collection for LeWM
    train_diamond_v2.sh     Paper-matched DIAMOND training
    runpod_train_day.sh     Master training script
  checkpoints/              Model weights (gitignored)
  data/                     Training data (gitignored)
  docs/
    AURA_PRD.md            This document
```

★ = key files

---

## 8. Key References

| Paper | Repo | Role |
|-------|------|------|
| LeWorldModel (Maes et al., 2026) | github.com/lucas-maes/le-wm | JEPA architecture + training |
| stable-worldmodel | github.com/galilai-group/stable-worldmodel | Env framework + evaluation |
| LeWM checkpoints | huggingface.co/collections/quentinll/lewm | Pretrained models |
| DIAMOND (Alonso et al., NeurIPS 2024) | github.com/eloialonso/diamond | Diffusion world model |
| EDM (Karras et al., 2022) | arxiv.org/abs/2206.00364 | Diffusion framework |

---

## 9. Constraints

- **Hardware**: M3 MacBook Air 16GB RAM — CPU inference only, no local training
- **Budget**: ~$10-20 RunPod per training session
- **JEPA runs locally** at ~15fps for game demos
- **DIAMOND needs GPU** for real-time (2fps on CPU)
- **No hybrid approaches**: world model must be 100% neural prediction/generation

---

*End of document — AURA PRD v0.2 — WAWE Studio — April 1, 2026*
