"""AURA Bounce — State dynamics world model inference.

State model predicts physics directly:
  (state_history + audio) → delta_state → next_state → render

No JEPA encoding, no probe, no lossy chain. Direct state prediction.
"""
import argparse
import json
import logging
import time
import asyncio
import random

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world_model.envs.bounce_world import BounceWorld
from world_model.ascii_model.state_model import StateDynamicsModel, STATE_DIM, HISTORY
from world_model.ascii_model.model import frame_to_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.infer_bounce")

_model = None
_device = None
_checkpoint_path = None
_state_mean = None
_state_std = None


def get_model():
    global _model, _device, _state_mean, _state_std
    if _model is not None:
        return _model, _device

    device = torch.device("cpu")
    _device = device
    logger.info("Loading from %s ...", _checkpoint_path)
    ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)

    model = StateDynamicsModel(hidden=ckpt.get("hidden", 128))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    _model = model

    _state_mean = ckpt["state_mean"]
    _state_std = ckpt["state_std"]

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded: %s params", f"{n_params:,}")
    return _model, _device


app = FastAPI(title="AURA Bounce — State World Model", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


def clamp_state(state_raw):
    """Keep state in valid physical bounds."""
    x, y, vx, vy, g = state_raw
    x = np.clip(x, 0.02, 0.98)   # stay on screen
    y = np.clip(y, 0.02, 0.95)   # stay between ceiling and floor
    vx = np.clip(vx, -0.5, 0.5)  # cap velocity
    vy = np.clip(vy, -0.8, 0.8)
    g = np.clip(g, 0.1, 1.0)     # gravity stays positive
    return np.array([x, y, vx, vy, g], dtype=np.float32)


@app.websocket("/ws-flappy")
async def flappy_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Flappy client connected")

    try:
        model, device = get_model()
    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    renderer = BounceWorld()
    renderer.reset(seed=42)

    # Initial state: ball at center
    state_raw = np.array([0.5, 0.5, 0.0, 0.0, 0.5], dtype=np.float32)
    state_norm = (torch.tensor(state_raw) - _state_mean) / _state_std

    # History buffer (normalized states)
    state_history = [state_norm.clone() for _ in range(HISTORY)]

    # Game state
    obstacles = []
    obstacle_timer = 0
    score = 0
    game_over = False
    frame_count = 0

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            audio_list = msg.get("audio")
            if audio_list is None or len(audio_list) != 16:
                continue

            audio_np = np.array(audio_list, dtype=np.float32)
            audio_t = torch.tensor([audio_list], dtype=torch.float32, device=device)

            # --- World model prediction ---
            with torch.no_grad():
                hist_t = torch.stack(state_history[-HISTORY:]).unsqueeze(0)  # (1, 3, 5)
                delta = model(hist_t, audio_t)  # (1, 5)

                # Apply delta to get next state (normalized)
                next_norm = state_history[-1] + delta[0]
                state_history.append(next_norm)
                if len(state_history) > HISTORY + 5:
                    state_history = state_history[-(HISTORY + 3):]

                # Denormalize
                state_raw = (next_norm * _state_std + _state_mean).numpy()
                state_raw = clamp_state(state_raw)

                # Re-normalize clamped state for next step's history
                state_history[-1] = (torch.tensor(state_raw) - _state_mean) / _state_std

            # --- Apply to renderer ---
            renderer.ball_x = float(state_raw[0] * renderer.W)
            renderer.ball_y = float(state_raw[1] * renderer.H)

            # Trail
            renderer.trail.append((renderer.ball_x, renderer.ball_y))
            if len(renderer.trail) > 10:
                renderer.trail = renderer.trail[-10:]

            # Particles on loud audio
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

            # --- Obstacles ---
            obstacle_timer += 1
            if obstacle_timer >= 25 and not game_over:
                obstacle_timer = 0
                gap_y = random.randint(8, renderer.H - 14)
                obstacles.append({"x": float(renderer.W - 2), "gap_y": gap_y,
                                   "gap_size": 10, "scored": False})

            alive_obs = []
            for obs in obstacles:
                obs["x"] -= 0.6
                if obs["x"] > -3:
                    alive_obs.append(obs)
                if not obs["scored"] and obs["x"] < renderer.ball_x:
                    obs["scored"] = True
                    score += 1
            obstacles = alive_obs

            # --- Collision ---
            ball_col = int(renderer.ball_x)
            ball_row = int(renderer.ball_y)
            hit = False
            for obs in obstacles:
                ox = int(obs["x"])
                if abs(ball_col - ox) <= 2:
                    if ball_row < obs["gap_y"] or ball_row > obs["gap_y"] + obs["gap_size"]:
                        hit = True

            if renderer.ball_y <= 2 or renderer.ball_y >= renderer.H - 4:
                hit = True

            if hit and not game_over:
                game_over = True

            # --- Reset on noise after game over ---
            if game_over and rms > 0.3:
                game_over = False
                state_raw = np.array([0.5, 0.5, 0.0, 0.0, 0.5], dtype=np.float32)
                state_norm = (torch.tensor(state_raw) - _state_mean) / _state_std
                state_history = [state_norm.clone() for _ in range(HISTORY)]
                renderer.ball_x = renderer.W / 2
                renderer.ball_y = renderer.H / 2
                obstacles = []
                score = 0

            # --- Render ---
            frame_str = renderer.render_ascii(audio_np)
            lines = [list(line) for line in frame_str.split('\n')]

            # Draw obstacles
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

            # Score / game over
            if game_over:
                msg_str = f" GAME OVER! Score: {score} — make noise to restart "
            else:
                msg_str = f" SCORE: {score} "
            mid = renderer.W // 2 - len(msg_str) // 2
            if len(lines) > 0:
                for si, ch in enumerate(msg_str):
                    if 0 <= mid + si < len(lines[0]):
                        lines[0][mid + si] = ch

            frame_str = '\n'.join(''.join(line) for line in lines)

            await ws.send_text(json.dumps({"frame": frame_str}))
            frame_count += 1
            if frame_count % 200 == 0:
                logger.info("Flappy: %d frames, score=%d", frame_count, score)

    except WebSocketDisconnect:
        logger.info("Client disconnected after %d frames", frame_count)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)


@app.websocket("/ws")
async def oneshot_endpoint(ws: WebSocket):
    """One-shot launch mode — kept for testing."""
    await ws.accept()
    # Minimal: just echo idle frames
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
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
    logger.info("Starting AURA Bounce (State World Model) on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
