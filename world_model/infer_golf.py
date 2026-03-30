"""AURA Golf — JEPA world model predicts ball trajectory from audio shot.

Paper-aligned: short-horizon rollout (15-20 steps), sliding window of 3,
audio = action, JEPA predicts trajectory in latent space, probe decodes state.

Game: shout → ball flies → lands near target → score.
"""
import argparse
import json
import logging
import time
import asyncio
import random
import math

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world_model.envs.bounce_world import BounceWorld
from world_model.ascii_model.jepa_proper import JEPAWorldModel, StateProbe, EMBED_DIM, HISTORY_SIZE
from world_model.ascii_model.model import frame_to_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.golf")

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
    logger.info("Loading %s ...", _checkpoint_path)
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
    else:
        _probe = None

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("JEPA Golf: %s params, probe=%s", f"{n_params:,}", _probe is not None)
    return _model, _probe, _device


app = FastAPI(title="AURA Golf — JEPA World Model", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "probe": _probe is not None}


TARGET_CHAR = "◎"
HOLE_CHAR = "⊕"
TEE_CHAR = "▫"


def render_golf_frame(renderer, target_x, target_y, trail=None, show_distance=False, score_text=""):
    """Render the golf course with ball, target, and trajectory trail."""
    frame_str = renderer.render_ascii(None)
    lines = [list(l) for l in frame_str.split('\n')]

    # Draw ground line
    ground_y = renderer.H - 3
    for x in range(1, renderer.W - 1):
        if ground_y < len(lines) and x < len(lines[ground_y]):
            lines[ground_y][x] = "─"

    # Draw tee
    tee_x, tee_y = int(renderer.W * 0.1), ground_y - 1
    if 0 <= tee_y < len(lines) and 0 <= tee_x < len(lines[tee_y]):
        lines[tee_y][tee_x] = TEE_CHAR

    # Draw target/hole
    tx, ty = int(target_x), int(target_y)
    if 0 <= ty < len(lines) and 0 <= tx < len(lines[ty]):
        lines[ty][tx] = TARGET_CHAR
    # Flag above hole
    if ty > 0 and 0 <= tx < len(lines[ty - 1]):
        lines[ty - 1][tx] = "▲"

    # Draw trajectory trail
    if trail:
        for i, (px, py) in enumerate(trail):
            ix, iy = int(px), int(py)
            if 1 <= ix < renderer.W - 1 and 1 <= iy < ground_y:
                alpha = i / max(len(trail), 1)
                ch = "·" if alpha < 0.5 else "∘"
                lines[iy][ix] = ch

    # Score text at top
    if score_text:
        mid = renderer.W // 2 - len(score_text) // 2
        for si, ch in enumerate(score_text):
            if 0 <= mid + si < len(lines[0]):
                lines[0][mid + si] = ch

    return '\n'.join(''.join(l) for l in lines)


@app.websocket("/ws-golf")
async def golf_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Golf client connected")

    try:
        model, probe, device = get_model()
    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    if probe is None:
        await ws.send_json({"error": "No probe in checkpoint"})
        await ws.close(code=1011)
        return

    renderer = BounceWorld()
    hole = 1
    strokes = 0
    total_score = 0

    # Generate first target
    ground_y = renderer.H - 4
    target_x = random.uniform(renderer.W * 0.4, renderer.W * 0.85)
    target_y = ground_y  # target on the ground

    # Place ball at tee
    renderer.reset(seed=42)
    renderer.ball_x = renderer.W * 0.1
    renderer.ball_y = ground_y
    renderer.vel_x = 0
    renderer.vel_y = 0

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mode = msg.get("mode", "idle")

            if mode == "idle":
                score_text = f" HOLE {hole} | STROKES: {strokes} | SCORE: {total_score} "
                frame = render_golf_frame(renderer, target_x, target_y,
                                           score_text=score_text)
                await ws.send_text(json.dumps({
                    "type": "ready", "frame": frame,
                    "hole": hole, "strokes": strokes
                }))

            elif mode == "shot":
                shot_audio = msg.get("audio", [0.0] * 16)
                strokes += 1
                rms = (shot_audio[12] + shot_audio[13]) / 2
                logger.info("SHOT! hole=%d stroke=%d rms=%.2f", hole, strokes, rms)

                # --- Seed frames: ball at tee ---
                renderer.ball_x = renderer.W * 0.1
                renderer.ball_y = ground_y
                renderer.vel_x = 0
                renderer.vel_y = 0
                renderer.trail = []

                seed_frames = []
                zero_audio = np.zeros(16, dtype=np.float32)
                shot_np = np.array(shot_audio, dtype=np.float32)

                for i in range(HISTORY_SIZE + 1):
                    audio = shot_np if i < 4 else zero_audio
                    renderer.step(audio)
                    frame_str = renderer.render_ascii(audio)
                    idx = frame_to_indices(frame_str)
                    if not isinstance(idx, torch.Tensor):
                        idx = torch.from_numpy(idx)
                    seed_frames.append(idx.long())

                # Send seed frames
                for i in range(len(seed_frames)):
                    score_text = f" HOLE {hole} | SWING! "
                    f = render_golf_frame(renderer, target_x, target_y,
                                           score_text=score_text)
                    await ws.send_text(json.dumps({
                        "type": "flight", "frame": f
                    }))
                    await asyncio.sleep(0.08)

                # --- JEPA rollout: predict trajectory ---
                ROLLOUT_STEPS = 20  # paper-aligned short horizon

                with torch.no_grad():
                    # Encode seed frames
                    seed_t = torch.stack(seed_frames).unsqueeze(0).to(device)
                    seed_emb = model.encode(seed_t)
                    emb_buffer = list(seed_emb[0].unbind(0))

                    # Audio embeddings: shot for first 3, zero for rest
                    shot_t = torch.tensor([shot_audio], dtype=torch.float32, device=device)
                    zero_t = torch.zeros(1, 16, device=device)
                    shot_emb = model.audio_encoder(shot_t)[0]
                    zero_emb = model.audio_encoder(zero_t)[0]
                    audio_emb_buffer = [shot_emb.clone() for _ in range(HISTORY_SIZE)]

                    trail = [(renderer.ball_x, renderer.ball_y)]
                    prev_x, prev_y = renderer.ball_x, renderer.ball_y
                    SMOOTH = 0.5

                    for step in range(ROLLOUT_STEPS):
                        # Sliding window of 3 (paper-exact)
                        ctx = torch.stack(emb_buffer[-HISTORY_SIZE:]).unsqueeze(0)
                        ctx_a = torch.stack(audio_emb_buffer[-HISTORY_SIZE:]).unsqueeze(0)

                        pred = model.predict_next(ctx, ctx_a)
                        emb_buffer.append(pred[0])

                        # Use zero audio after first 3 steps (ball in flight)
                        audio_emb_buffer.append(zero_emb.clone())

                        # Trim
                        if len(emb_buffer) > HISTORY_SIZE + 10:
                            emb_buffer = emb_buffer[-(HISTORY_SIZE + 5):]
                        if len(audio_emb_buffer) > HISTORY_SIZE + 10:
                            audio_emb_buffer = audio_emb_buffer[-(HISTORY_SIZE + 5):]

                        # Decode state
                        state_norm = probe(pred)
                        state = (state_norm[0] * _state_std + _state_mean).cpu().numpy()

                        raw_x = float(state[0] * renderer.W)
                        raw_y = float(state[1] * renderer.H)
                        smooth_x = SMOOTH * prev_x + (1 - SMOOTH) * raw_x
                        smooth_y = SMOOTH * prev_y + (1 - SMOOTH) * raw_y
                        prev_x, prev_y = smooth_x, smooth_y

                        renderer.ball_x = float(np.clip(smooth_x, 1, renderer.W - 2))
                        renderer.ball_y = float(np.clip(smooth_y, 1, ground_y))

                        trail.append((renderer.ball_x, renderer.ball_y))
                        renderer.trail = trail[-10:]
                        renderer.particles = []

                        # Render and send
                        score_text = f" HOLE {hole} | FLIGHT step {step+1}/{ROLLOUT_STEPS} "
                        f = render_golf_frame(renderer, target_x, target_y,
                                               trail=trail, score_text=score_text)
                        await ws.send_text(json.dumps({
                            "type": "flight", "frame": f
                        }))
                        await asyncio.sleep(0.1)

                        # Check for early landing (ball reached ground)
                        if renderer.ball_y >= ground_y - 1:
                            renderer.ball_y = ground_y
                            break

                # --- Ball landed ---
                dist = math.sqrt((renderer.ball_x - target_x)**2 +
                                  (renderer.ball_y - target_y)**2)

                # Scoring
                if dist < 3:
                    points = 3  # hole in one area
                    msg_extra = "AMAZING!"
                elif dist < 8:
                    points = 2
                    msg_extra = "Great shot!"
                elif dist < 15:
                    points = 1
                    msg_extra = "Good"
                else:
                    points = 0
                    msg_extra = "Try again"

                total_score += points

                score_text = f" {msg_extra} Dist: {dist:.1f} | +{points}pts | Total: {total_score} "
                f = render_golf_frame(renderer, target_x, target_y,
                                       trail=trail, score_text=score_text)
                await ws.send_text(json.dumps({
                    "type": "landed", "frame": f,
                    "distance": dist, "score": total_score,
                    "points": points
                }))

                # Next hole after a pause
                await asyncio.sleep(2.0)
                hole += 1
                target_x = random.uniform(renderer.W * 0.3, renderer.W * 0.9)
                target_y = ground_y
                renderer.ball_x = renderer.W * 0.1
                renderer.ball_y = ground_y
                renderer.vel_x = 0
                renderer.vel_y = 0
                renderer.trail = []

                score_text = f" HOLE {hole} | STROKES: {strokes} | SCORE: {total_score} "
                f = render_golf_frame(renderer, target_x, target_y,
                                       score_text=score_text)
                await ws.send_text(json.dumps({
                    "type": "ready", "frame": f,
                    "hole": hole, "strokes": strokes
                }))

    except WebSocketDisconnect:
        logger.info("Golf client disconnected")
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)


def main():
    global _checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    _checkpoint_path = args.checkpoint

    import uvicorn
    logger.info("Starting AURA Golf on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
