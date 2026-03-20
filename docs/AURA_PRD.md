# AURA — Audio-Conditioned World Model
## Product Requirements Document v0.1
**WAWE Studio** · March 2026 · Internal

---

## Document Info

| Field | Value |
|-------|-------|
| Project | AURA — Audio-conditioned World Model |
| Studio | WAWE Studio (Chile) |
| Status | Pre-development — ready for Claude Code |
| Target | Playable browser demo + trained world model |
| Deadline | April 14, 2026 (2 weeks to demo; April 30 competition) |
| Compute | Google Colab / RunPod (no local GPU) |
| Multiplayer | Yes — WebSocket event sync from V1 |
| Open Source | Training pipeline: yes. Model weights: TBD |

---

## 1. Vision

AURA is an audio-conditioned world model that generates and simulates a physically consistent, navigable 3D alien environment in real time. The model does not use a traditional game engine. A neural network IS the engine — it learns the physics, spatial structure, and dynamics of the world, then predicts future states frame by frame.

**What makes AURA different:**

- Audio is not a visual effect layer — it is a continuous physics parameter that modifies the world model's learned dynamics at the recurrent state level
- Loud bass → organic structures grow. Fast tempo → camera velocity increases in latent space. Spectral density → world complexity and branching increases
- The world model maintains spatial consistency, object permanence, and navigable 3D space — not generated video, but a learned simulation
- Runs locally on consumer hardware. No cloud inference required for gameplay
- Multiplayer: all players share the same audio stream as a physics driver. Server syncs events only; each client runs the world model locally

---

## 2. Goals & Non-Goals

### 2.1 Goals for V1

- Train a context-conditioned world model (cRSSM based on DreamerV3) where audio features serve as the context vector modifying world dynamics
- Generate synthetic training data from a custom minimal Python simulation of an alien corridor environment with procedural visual style
- Produce a playable browser demo: rail navigation (forward camera) through the world model's generated frames, with audio driving physical changes in real time
- Basic DDR/rhythm mechanic: player input events must be synchronized to music beats; world reacts visually and physically on hit/miss
- WebSocket multiplayer: multiple players navigate the same world simultaneously; server holds authoritative audio state and event log
- Training runs on RunPod or Google Colab (no local GPU dependency)

### 2.2 Non-Goals for V1

- Photorealistic visuals — stylized alien-organic aesthetic is intentional and reduces model complexity
- Free roam or open world — rail navigation only in V1
- General-purpose world model — AURA is domain-specific to this constrained environment
- Mobile support — desktop browser only for V1
- Flash memory inference optimization (LLM-in-a-Flash techniques) — documented as future work in Appendix B

---

## 3. Architecture

### 3.1 System Overview

AURA consists of three loosely coupled systems:

**System 1 — World Model (Python, trained offline, runs inference locally)**

A context-conditioned Recurrent State-Space Model (cRSSM) forked from DreamerV3. Audio features are injected as a context vector `c_t` into the RSSM's recurrent transition function alongside action inputs. The model predicts the next world state in latent space and decodes it to pixel-space frames.

**System 2 — Browser Client (Three.js, Web Audio API, WebSockets)**

Receives decoded frames from the world model inference server, renders them in Three.js with a forward-moving camera. Web Audio API provides real-time FFT analysis of audio input. Player input is captured and sent to the server as timestamped events.

**System 3 — Event Server (Node.js, WebSocket)**

Holds authoritative audio state (current track position, BPM, FFT snapshot) and broadcasts it to all connected clients. Receives player events and logs them for replay and scoring. Does NOT run the world model — each client runs its own inference instance.

---

### 3.2 World Model Architecture

**Base:** DreamerV3 (Hafner et al., Nature 2025) — open source, single GPU trainable, latent space world model.

**Modification:** cRSSM (context-conditioned RSSM, from "Dreaming of Many Worlds", RLC 2024). Audio feature vector replaces the "difficulty context" used in the original paper. This is architecturally clean — the context slot already exists in the design.

**Recurrent state transition:**

```
h_t = f(h_{t-1}, z_{t-1}, a_t, c_t)
z_t = encoder(obs_t)
obs_{t+1} = decoder(h_t, z_t)

Where:
  h_t  = deterministic recurrent state (GRU hidden)
  z_t  = stochastic categorical latent (world state)
  a_t  = action (camera forward velocity, left/right)
  c_t  = audio context vector (FFT bins, onset, BPM, energy)
```

---

### 3.3 Audio Feature Vector

Extracted per frame (~10-30fps) using Web Audio API (browser) or librosa (training):

| Feature | Source | World Effect |
|---------|--------|--------------|
| Sub-bass energy (20–80 Hz) | FFT | Growth/density of organic structures |
| Mid energy (250 Hz–2 kHz) | FFT | Color saturation and light emission |
| High frequency energy (4–20 kHz) | FFT | Particle/detail complexity |
| Onset detection | Peak detection | Discrete physics events (spawn, collapse) |
| Estimated BPM | Autocorrelation | Camera forward velocity |
| Spectral centroid | FFT | World temperature (warm/cool palette) |
| RMS energy | RMS | Overall world activity level |

**Vector:** 16–32 floats, normalized to [0, 1], concatenated with action vector before RSSM input.

---

### 3.4 Technology Stack

| Layer | Technology | Role |
|-------|-----------|------|
| World Model | DreamerV3 fork (PyTorch/JAX) | Core neural simulator — cRSSM with audio context |
| Training | RunPod / Google Colab | H100 or A100, pay-per-run, ~$20–50 per run |
| Sim / Data Gen | Custom Python (NumPy) | Minimal corridor sim, no physics engine needed |
| Audio Analysis | librosa (training), Web Audio API (inference) | FFT, onset, BPM, spectral features |
| 3D Rendering | Three.js | Displays decoded world model frames, camera path |
| Audio Input | Web Audio API | Microphone + pre-loaded tracks, both supported |
| Multiplayer | Node.js + WebSocket (ws) | Event sync, audio state broadcast, player sessions |
| Client Deploy | Vite + vanilla JS | Fast dev, zero framework overhead |
| Model Export | ONNX (V2) | Browser-native inference future |

---

### 3.5 Decoder Upgrade Path: CNN → Diffusion

The V1 CNN decoder produces blurry 64x64 output due to MSE loss averaging. For a navigable world where the neural output IS the rendered environment, visual quality must improve. Three upgrade tiers:

**Tier 1 — CNN 128x128 (current upgrade, T4-compatible)**
- Add 5th conv/deconv layer: 128→64→32→16→8→4
- Config: `aura_128` with `encoder_channels: [32, 64, 128, 256, 512]`
- 4x more pixels, same architecture. Sharper but still soft edges.
- Training: ~2x slower than 64x64 on T4

**Tier 2 — Diffusion Decoder (DIAMOND-style, target for demo)**
- Keep cRSSM for dynamics + audio conditioning (it's great at this)
- Replace CNN decoder with a small DDPM conditioned on RSSM features
- 10-20 denoising steps → sharp 128x128 or 256x256 frames
- Reference: DIAMOND (Alonso et al., 2024) — diffusion world model for Atari/CS:GO
- Reference: GameNGen (Google, 2024) — real-time DOOM via diffusion world model
- Inference: ~50-100ms/frame (acceptable at 10-15fps for rail navigation)
- Training: needs A100, ~4-8 hours for 256x256

**Tier 3 — VAE-GAN Decoder (alternative)**
- Add discriminator network that penalizes blurry output
- Faster than diffusion (~5-10ms/frame) but harder to train (GAN instability)
- Good fallback if diffusion is too slow for real-time

**Architecture diagram (Tier 2):**
```
Audio → FFT → context c_t
                    ↓
cRSSM: h_t = f(h_{t-1}, z_{t-1}, a_t, c_t)   ← dynamics (unchanged)
                    ↓
            features = [h_t, z_t]
                    ↓
    Diffusion Decoder (conditioned on features)  ← replaces CNN
            noise → denoise × 15 steps
                    ↓
            256×256 RGB frame                    ← navigable output
```

**Decision:** Start with Tier 1 (128x128 CNN) for immediate improvement. Implement Tier 2 (diffusion decoder) before April 14 demo if 128x128 CNN quality isn't sufficient for navigation.

---

## 4. Training Pipeline

### 4.1 Synthetic Data Generation

We generate training data entirely synthetically — no real-world footage required.

Each sample tuple: `(obs_t, action_t, audio_ctx_t, obs_{t+1})`

**Simulation design:**

The sim is a minimal Python script generating corridor sections as nested noise-displaced tubes with bioluminescent organic nodes. Audio features are computed from a synthetic procedural audio track. Physics parameters are driven by audio features via manually defined rules (e.g., `bass_energy * 3 = node_size_multiplier`). The world model learns to generalize these correlations, producing emergent behavior beyond the scripted rules.

**Output per frame:**
- 64×64 RGB egocentric observation
- Action label (forward / turn-left / turn-right)
- 16-float audio feature vector
- Physics state metadata (for validation only)

**Dataset size targets:**
- 50K frames — minimum for first training run
- 500K frames — target for playable quality
- Generation rate: ~1,000 frames/min on CPU → 500K frames in ~8 hours (free Colab)

### 4.2 Training Configuration

```yaml
base_config: dreamerv3/size50m
modifications:
  - add audio_context input (16 floats) to cRSSM transition function
  - audio_context_rank: 32  # cRSSM conditioning rank
observation: 64x64 RGB
action_space: 3 discrete (forward, left, right)
context_dim: 16
batch_size: 16
sequence_length: 64
training_steps: 100_000
estimated_time: 6–8 hours on A100
estimated_cost: $12–16 on RunPod
```

### 4.3 Validation Criteria

A passing model must demonstrate:
- Visible world state change within 3 frames of an audio event
- Spatial consistency across 50+ consecutive predicted frames without drift
- Correct temporal ordering: onset → structure spawn, silence → world dimming

---

## 5. Game Mechanic

### 5.1 Core Loop

AURA V1 is a rail experience with a rhythm layer.

1. Audio stream drives continuous world physics (growth, speed, light)
2. On detected beat onset: interactive node spawns in player's path
3. Player clicks node within timing window → hit → world pulse effect, score up
4. Missed node → world dims briefly, velocity slows
5. World state continuously evolves — no two runs identical

### 5.2 Audio Input Modes

- **Pre-loaded track:** curated alien ambient / electronic music packaged with demo
- **Microphone:** live audio input analyzed in real time — any sound drives the world
- Mode selector in UI before session start

### 5.3 Multiplayer

All players in a session share the same audio state. Server broadcasts current audio feature vector to all clients at ~30Hz. Each client runs its own world model inference. World states may diverge between clients based on player actions — this is intentional. Server reconciles discrete events (node hits, spawns) but not continuous world state.

---

## 6. Development Phases

| Phase | Name | Duration | Key Output |
|-------|------|----------|------------|
| P0 | Foundation | 3 days | Sim runs, data pipeline, DreamerV3 fork with audio input |
| P1 | First Training Run | 2 days | Trained model, qualitative validation passing |
| P2 | Browser Demo | 4 days | Playable rail demo, audio driving world in real time |
| P3 | Rhythm Mechanic | 3 days | Beat detection, interactive nodes, hit/miss scoring |
| P4 | Multiplayer | 3 days | WebSocket server, multi-client audio sync |
| P5 | Polish + Deploy | 3 days | Visual pass, perf optimization, public URL |

**Total: 18 days → target demo April 14**

### P0 — Foundation (Days 1–3)
- [ ] Fork DreamerV3, strip unused domains, add audio context input to cRSSM
- [ ] Write minimal Python corridor sim with procedural audio + frame logger
- [ ] Generate 50K frame dataset on Colab CPU
- [ ] Set up RunPod training script with wandb logging
- [ ] Validate data pipeline: confirm `(obs, action, audio_ctx)` tuples load correctly

### P1 — First Training Run (Days 4–5)
- [ ] Launch training on RunPod A100 — size50m, 100K steps (~$12)
- [ ] Monitor reconstruction loss + KL divergence via wandb
- [ ] Qualitative eval: play audio, observe predicted frames, confirm audio→world correlation
- [ ] If correlation weak: adjust normalization, increase cRSSM conditioning rank
- [ ] Save checkpoint as `aura-v0.1`

### P2 — Browser Demo (Days 6–9)
- [ ] Set up Vite project with Three.js
- [ ] Build inference server: FastAPI endpoint (audio_ctx + action → next frame)
- [ ] Connect browser client to inference server via WebSocket
- [ ] Implement Web Audio API FFT pipeline, map to audio context vector
- [ ] Render world model frames in Three.js on forward-moving camera path
- [ ] Support both audio modes: pre-loaded track + microphone

### P3 — Rhythm Mechanic (Days 10–12)
- [ ] Implement beat onset detector (autocorrelation + peak detection)
- [ ] Spawn interactive nodes in Three.js at beat-synchronized positions
- [ ] Hit detection via Three.js raycasting on click/tap
- [ ] Wire hit/miss to world model context (hit → increase energy, miss → decrease)
- [ ] Score overlay (minimal, in-world aesthetic)

### P4 — Multiplayer (Days 13–15)
- [ ] Node.js WebSocket server: session management, audio state broadcast at 30Hz
- [ ] Client receives audio state override from server (replaces local FFT)
- [ ] Server tracks discrete events: node spawns, player hits, session score
- [ ] Session URL sharing: `wawe.app/session/[id]`

### P5 — Polish + Deploy (Days 16–18)
- [ ] Visual style pass: alien color palette, emission maps, depth fog
- [ ] Profile inference latency, optimize to maintain >15fps
- [ ] Deploy inference server to RunPod persistent instance or Hetzner VPS
- [ ] Deploy browser client to Vercel or Netlify
- [ ] Record demo video

---

## 7. Repository Structure

```
aura/
  world_model/           ← Python: DreamerV3 fork
    dreamer/
      rssm.py            ← cRSSM with audio context conditioning
      encoder.py         ← CNN obs encoder
      decoder.py         ← CNN obs decoder
      agent.py           ← Actor-critic
      audio.py           ← Audio feature extraction (librosa)
    sim/
      corridor.py        ← Procedural alien corridor generator
      audio_gen.py       ← Synthetic audio + feature logging
      dataset.py         ← Data generation entry point
    train.py             ← Training entry point
    infer.py             ← Inference server (FastAPI)
    requirements.txt
  client/                ← Browser: Three.js
    src/
      audio.js           ← Web Audio API, FFT, beat detection
      world.js           ← Three.js scene, camera path, node rendering
      net.js             ← WebSocket client, audio sync
      game.js            ← Hit detection, scoring, game state
      main.js            ← Entry point
    index.html
    vite.config.js
    package.json
  server/                ← Node.js: WebSocket event server
    index.js
    session.js           ← Session management, audio broadcast
    package.json
  data/                  ← Generated training data (gitignored)
  checkpoints/           ← Model weights (gitignored)
  .env.example
  README.md
```

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Audio conditioning too weak (model ignores audio context) | Medium | Add auxiliary audio-prediction loss; increase cRSSM rank; curriculum training |
| Inference too slow for real-time (>100ms per frame) | Medium | Run inference server-side; reduce to 64×64; cache latent states |
| World model drifts after 50+ frames | Medium | Longer training sequences (64+ rollout); R2I memory augmentation |
| Synthetic data too simple — model learns nothing | Low | Iterate sim visuals first; validate dataset with human inspection before training |
| RunPod costs exceed budget | Low | Use Colab for prototyping; only pay for full training runs |

---

## 9. Parameter Golf Competition (Optional)

OpenAI's Parameter Golf (March 18 – April 30, 2026): train the best LM within 16MB artifact + 10-minute 8×H100 budget.

After AURA V1 ships (~April 14), architecture insights map directly:
- cRSSM context conditioning → parameter-efficient conditioning
- Activation sparsity via FATReLU → reduces effective parameter count
- Compact latent design → fits 16MB with quantization
- Synthetic curriculum → efficient pretraining

**Scope:** 2–3 days post-V1. Treat as funded research feeding back into AURA V2.

Repo: https://github.com/openai/parameter-golf

---

## Appendix A — Key Papers

| Paper | Where to find | Why it matters |
|-------|--------------|----------------|
| DreamerV3 (Hafner et al., Nature 2025) | github.com/danijar/dreamerv3 | Base architecture — fork this |
| Dreaming of Many Worlds / cRSSM (RLC 2024) | arxiv.org/pdf/2309.xxxx | Audio conditioning mechanism |
| DIAMOND (Alonso et al., NeurIPS 2024) | github.com/eloialonso/diamond | Diffusion WM reference |
| Navigation World Models (Bar et al., CVPR 2025) | openaccess.thecvf.com | 3D egocentric navigation in WM |
| V-JEPA 2 (Meta, 2025) | github.com/facebookresearch/jepa | Latent prediction reference |

---

## Appendix B — Future: LLM-in-a-Flash for Local Inference

> **Status: V2 only. Not in scope for V1.**

Based on "LLM in a Flash" (Alizadeh et al., Apple, arXiv 2312.11514).

**Core idea:** RSSM FFN layers exhibit high activation sparsity analogous to the 90–97% sparsity in LLM FFN layers. By storing model weights in SSD and loading only predicted-active neurons per frame, we can run a world model larger than available DRAM.

**Applicable techniques for V2:**
- **FATReLU activations** in RSSM FFN blocks → induce >85% sparsity with minimal performance loss
- **Low-rank activation predictor** per FFN layer → forecasts which neurons activate next frame, enables selective weight loading
- **Sliding window cache** → retain active weights from last k frames in DRAM, load only delta per new frame (high neuron reuse across consecutive frames makes this very efficient)
- **Row-column bundling** → store paired up/down projection weights together for 2× flash read throughput

**Why V1 doesn't need this:**
V1 uses a persistent RunPod inference server. Local inference optimization unblocks after the model architecture is validated and audio conditioning is confirmed working.

**V2 target:** ONNX export + selective loading → full world model inference in browser via WebGPU, zero server dependency.

---

*End of document — AURA PRD v0.1 — WAWE Studio*
