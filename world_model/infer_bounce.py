"""AURA Bounce — JEPA world model inference (LeWM paper-aligned).

JEPA predicts latent dynamics → state probe extracts ball physics →
BounceWorld renders clean ASCII.

Two modes:
  /ws-flappy: continuous audio, JEPA predicts each step
  /ws: one-shot launch
"""
import argparse
import json
import logging
import time
import asyncio
import random

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world_model.envs.bounce_world import BounceWorld
from world_model.ascii_model.jepa_proper import JEPAWorldModel, StateProbe, EMBED_DIM, HISTORY_SIZE
from world_model.ascii_model.model import frame_to_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.infer_bounce")

_model = None
_probe = None
_device = None
_checkpoint_path = None
_state_mean = None
_state_std = None


def get_model():
    global _model, _probe, _device, _state_mean, _state_std
    if _model is not None:
        return _model, _probe, _device

    device = torch.device("cpu")
    _device = device
    logger.info("Loading from %s ...", _checkpoint_path)
    ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)

    embed_dim = ckpt.get("embed_dim", EMBED_DIM)
    model = JEPAWorldModel(embed_dim=embed_dim)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    _model = model

    if "probe" in ckpt:
        probe_sd = ckpt["probe"]
        first_key = list(probe_sd.keys())[0]
        if first_key.startswith("net."):
            probe = StateProbe(embed_dim=embed_dim)
        else:
            probe = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 5)
            )
        probe.load_state_dict(probe_sd)
        probe = probe.to(device).eval()
        _probe = probe
        _state_mean = ckpt["state_mean"].to(device)
        _state_std = ckpt["state_std"].to(device)
        logger.info("Probe loaded")
    else:
        _probe = None
        logger.warning("No probe in checkpoint")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("JEPA: %s params, embed_dim=%d", f"{n_params:,}", embed_dim)
    return _model, _probe, _device


app = FastAPI(title="AURA Bounce — JEPA World Model", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "probe": _probe is not None}


def decode_state(probe, pred_emb):
    """Probe → denormalized state."""
    state_norm = probe(pred_emb)
    state = state_norm[0] * _state_std + _state_mean
    return state.cpu().numpy()


@app.websocket("/ws-flappy")
async def flappy_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Flappy connected")

    try:
        model, probe, device = get_model()
    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    if probe is None:
        await ws.send_json({"error": "No probe"})
        await ws.close(code=1011)
        return

    renderer = BounceWorld()
    renderer.reset(seed=42)
    renderer.ball_x = renderer.W / 2
    renderer.ball_y = renderer.H / 2

    # Encode initial frames into latent buffer
    init_audio = np.zeros(16, dtype=np.float32)
    seed_frames = []
    for _ in range(HISTORY_SIZE):
        frame_str = renderer.render_ascii(init_audio)
        idx = frame_to_indices(frame_str)
        if not isinstance(idx, torch.Tensor):
            idx = torch.from_numpy(idx)
        seed_frames.append(idx.long())

    with torch.no_grad():
        seed_t = torch.stack(seed_frames).unsqueeze(0).to(device)
        seed_emb = model.encode(seed_t)
        emb_buffer = list(seed_emb[0].unbind(0))

        zero_a = model.audio_encoder(torch.zeros(1, 16, device=device))[0]
        audio_emb_buffer = [zero_a.clone() for _ in range(HISTORY_SIZE)]

    # Game state
    obstacles = []
    obstacle_timer = 0
    score = 0
    game_over = False
    frame_count = 0
    smooth_x = renderer.W / 2
    smooth_y = renderer.H / 2
    SMOOTH = 0.7  # higher = smoother, less jitter

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            audio_list = msg.get("audio")
            if audio_list is None or len(audio_list) != 16:
                continue

            audio_np = np.array(audio_list, dtype=np.float32)
            audio_t = torch.tensor([audio_list], dtype=torch.float32, device=device)

            with torch.no_grad():
                # Encode audio
                audio_emb = model.audio_encoder(audio_t)[0]
                audio_emb_buffer.append(audio_emb)

                # JEPA: sliding window of 3 latents + 3 audio embeddings
                ctx = torch.stack(emb_buffer[-HISTORY_SIZE:]).unsqueeze(0)
                ctx_a = torch.stack(audio_emb_buffer[-HISTORY_SIZE:]).unsqueeze(0)

                # Predict next latent
                pred_emb = model.predict_next(ctx, ctx_a)
                emb_buffer.append(pred_emb[0])

                # Trim buffers
                if len(emb_buffer) > HISTORY_SIZE + 10:
                    emb_buffer = emb_buffer[-(HISTORY_SIZE + 5):]
                if len(audio_emb_buffer) > HISTORY_SIZE + 10:
                    audio_emb_buffer = audio_emb_buffer[-(HISTORY_SIZE + 5):]

                # Decode state
                state = decode_state(probe, pred_emb)

            # Smooth Y only — X is fixed (flappy bird style)
            raw_y = float(state[1] * renderer.H)
            smooth_y = SMOOTH * smooth_y + (1 - SMOOTH) * raw_y

            renderer.ball_x = 15.0  # fixed x position, like flappy bird
            renderer.ball_y = float(np.clip(smooth_y, 2, renderer.H - 5))

            # Trail + particles
            renderer.trail.append((renderer.ball_x, renderer.ball_y))
            if len(renderer.trail) > 10:
                renderer.trail = renderer.trail[-10:]

            rms = (audio_np[12] + audio_np[13]) / 2
            onset = (audio_np[6] + audio_np[7]) / 2
            renderer.particles = []
            if onset > 0.3 or rms > 0.5:
                for _ in range(int(max(onset, rms) * 5)):
                    renderer.particles.append({
                        'x': renderer.ball_x + renderer.rng.uniform(-2, 2),
                        'y': renderer.ball_y,
                        'vx': renderer.rng.uniform(-2, 2),
                        'vy': renderer.rng.uniform(-3, 0),
                        'life': renderer.rng.integers(2, 5),
                    })

            # Obstacles
            obstacle_timer += 1
            if obstacle_timer >= 25 and not game_over:
                obstacle_timer = 0
                gap_y = random.randint(8, renderer.H - 14)
                obstacles.append({"x": float(renderer.W - 2), "gap_y": gap_y, "gap_size": 10, "scored": False})

            alive = []
            for obs in obstacles:
                obs["x"] -= 0.6
                if obs["x"] > -3:
                    alive.append(obs)
                if not obs["scored"] and obs["x"] < renderer.ball_x:
                    obs["scored"] = True
                    score += 1
            obstacles = alive

            # Collision — strict: ball at x=15, pipes are 2 chars wide
            ball_row = int(renderer.ball_y)
            hit = renderer.ball_y <= 2 or renderer.ball_y >= renderer.H - 5
            for obs in obstacles:
                ox = int(obs["x"])
                # Pipe occupies columns ox and ox+1
                if 14 <= ox <= 17:  # ball is at x=15, check overlap
                    gy, gs = obs["gap_y"], obs["gap_size"]
                    if ball_row < gy or ball_row > gy + gs:
                        hit = True
                        break

            if hit and not game_over:
                game_over = True

            # Reset
            if game_over and rms > 0.3:
                game_over = False
                smooth_x = renderer.W / 2
                smooth_y = renderer.H / 2
                obstacles = []
                score = 0
                # Re-encode reset
                renderer.ball_x = renderer.W / 2
                renderer.ball_y = renderer.H / 2
                with torch.no_grad():
                    rf = renderer.render_ascii(None)
                    ridx = frame_to_indices(rf)
                    if not isinstance(ridx, torch.Tensor):
                        ridx = torch.from_numpy(ridx)
                    re = model.encode(ridx.long().unsqueeze(0).unsqueeze(0).to(device))
                    emb_buffer = [re[0, 0]] * HISTORY_SIZE

            # Render
            frame_str = renderer.render_ascii(audio_np)
            lines = [list(l) for l in frame_str.split('\n')]

            for obs in obstacles:
                ox = int(obs["x"])
                if ox < 1 or ox >= renderer.W - 1:
                    continue
                gy, gs = obs["gap_y"], obs["gap_size"]
                for y in range(1, renderer.H - 2):
                    if y < gy or y > gy + gs:
                        if 0 <= y < len(lines) and 0 <= ox < len(lines[y]):
                            lines[y][ox] = "█"
                        if ox + 1 < renderer.W - 1 and 0 <= y < len(lines) and ox + 1 < len(lines[y]):
                            lines[y][ox + 1] = "▓"

            msg_str = f" GAME OVER! Score: {score} — make noise to restart " if game_over else f" SCORE: {score} "
            mid = renderer.W // 2 - len(msg_str) // 2
            if lines:
                for si, ch in enumerate(msg_str):
                    if 0 <= mid + si < len(lines[0]):
                        lines[0][mid + si] = ch

            frame_str = '\n'.join(''.join(l) for l in lines)
            await ws.send_text(json.dumps({"frame": frame_str}))

            frame_count += 1
            if frame_count % 200 == 0:
                logger.info("Flappy: %d frames, score=%d", frame_count, score)

    except WebSocketDisconnect:
        logger.info("Disconnected after %d frames", frame_count)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)


@app.websocket("/ws")
async def oneshot_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            renderer = BounceWorld()
            renderer.ball_x = 40
            renderer.ball_y = 36
            frame = renderer.render_ascii(np.zeros(16, dtype=np.float32))
            await ws.send_text(json.dumps({"type": "idle", "frame": frame}))
    except WebSocketDisconnect:
        pass


def main():
    global _checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    _checkpoint_path = args.checkpoint

    import uvicorn
    logger.info("Starting AURA Bounce JEPA on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
