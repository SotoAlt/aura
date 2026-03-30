# AURA Development Roadmap

**Target**: Audio-reactive world model MVP demo
**Deadline**: April 14, 2026 (demo) / April 30, 2026 (competition)
**Start**: March 19, 2026

---

## Phase 0 — Foundation (Days 1-3)

- [ ] **P0.1** Vendor DreamerV3 source into `world_model/dreamer/` and `world_model/embodied/`
- [ ] **P0.2** Apply cRSSM audio context modifications to `rssm.py` + `agent.py` ← depends on P0.1
- [ ] **P0.3** Build corridor Gymnasium environment (`envs/corridor.py`) — can run in parallel with P0.1-P0.2
- [ ] **P0.4** Build audio feature extraction pipeline (`audio/features.py`, `audio/synthetic.py`) — can run in parallel
- [ ] **P0.5** Wire data generation pipeline + generate 100 test episodes locally ← depends on P0.2, P0.3, P0.4

**Milestone**: `python world_model/dreamer/main.py --configs aura --jax.platform cpu --run.steps 10` runs without errors

---

## Phase 1 — Training (Days 4-7)

- [ ] **P1.1** Create Colab data generation notebook → generate 5K-10K episodes on Drive ← depends on P0.5
- [ ] **P1.2** Create Colab training notebook → first training run (100K steps, size50m) ← depends on P1.1
- [ ] **P1.3** Create evaluation notebook → validate audio→world correlation with GIF rollouts ← depends on P1.2

**Milestone**: Trained model generates corridor frames that visibly respond to audio context changes

---

## Phase 2 — Browser Demo (Days 8-12)

- [ ] **P2.1** Build FastAPI inference server (`infer.py`) — WebSocket, CPU inference ← depends on P1.2
- [ ] **P2.2** Build Three.js client with Web Audio API — audio capture + scene rendering ← depends on P0.4
- [ ] **P2.3** Integration test — end-to-end audio-reactive demo ← depends on P2.1, P2.2

**Milestone**: Play music in browser → world model generates audio-reactive corridor in real time

---

## Deferred (post-MVP)

- [ ] **P3** Rhythm mechanic — beat detection, interactive nodes, hit/miss scoring
- [ ] **P4** Multiplayer — WebSocket event server, multi-client audio sync
- [ ] **P5** Polish — visual style pass, deploy to waweapps.win, demo video
- [ ] **Future** ONNX export + browser-native inference (WebGPU), LLM-in-a-Flash techniques

---

## Dependency Graph

```
P0.1 (Vendor DreamerV3)
  └→ P0.2 (cRSSM mods)
       └→ P0.5 (Data pipeline) ←── P0.3 (Corridor env)
            │                  ←── P0.4 (Audio features)
            └→ P1.1 (Colab data gen)
                 └→ P1.2 (Training)
                      ├→ P1.3 (Evaluation)
                      └→ P2.1 (Inference server)
                           └→ P2.3 (E2E demo) ←── P2.2 (Three.js client)
                                                    ↑
                                               P0.4 (Audio features)
```

## Parallel Work Opportunities

Tasks that can be worked on simultaneously:
- **P0.1** + **P0.3** + **P0.4** — vendor DreamerV3, build corridor env, build audio pipeline (all independent)
- **P2.1** + **P2.2** — inference server and client can be built in parallel once training is done
- **P1.3** + **P2.2** — evaluation and client dev can overlap
