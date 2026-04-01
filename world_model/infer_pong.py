"""Pong inference — JEPA predicts ball position, shown as ghost ball.

Real physics runs on client. Server receives state every 5 frames,
JEPA predicts 1-step ahead (where ball will be next), sends back
ghost position. Client renders ghost ball as gold overlay.

Paper-aligned: 1-step prediction from re-encoded frames.
"""
import argparse
import json
import logging
import asyncio

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world_model.ascii_model.jepa_pool import JEPAPool, StateProbe, EMBED_DIM, HISTORY_SIZE
from world_model.envs.pong_world import PongWorld

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.pong")

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
    logger.info("Loading %s", _checkpoint_path)
    ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)

    model = JEPAPool(embed_dim=ckpt.get("embed_dim", EMBED_DIM))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    _model = model

    if "probe" in ckpt:
        probe_sd = ckpt["probe"]
        out_dim = list(probe_sd.values())[-2].shape[0]  # last weight's output
        probe = StateProbe(embed_dim=ckpt.get("embed_dim", EMBED_DIM), state_dim=out_dim)
        try:
            probe.load_state_dict(probe_sd)
        except RuntimeError:
            # Fallback: build matching Sequential
            layers = []
            keys = [k for k in probe_sd if "weight" in k]
            for k in keys:
                in_f, out_f = probe_sd[k].shape[1], probe_sd[k].shape[0]
                layers.append(nn.Linear(in_f, out_f))
                if k != keys[-1]:
                    layers.append(nn.ReLU())
                    if f"{k.rsplit('.', 1)[0]}.{int(k.rsplit('.', 1)[1].split('w')[0]) + 1}.p" in str(probe_sd.keys()):
                        layers.append(nn.Dropout(0.1))
            probe = nn.Sequential(*layers)
            probe.load_state_dict(probe_sd)
        probe = probe.to(device).eval()
        _probe = probe
        _state_mean = ckpt.get("state_mean", torch.zeros(10)).to(device)
        _state_std = ckpt.get("state_std", torch.ones(10)).to(device)
    else:
        _probe = None

    n = sum(p.numel() for p in model.parameters())
    logger.info("Loaded: %s params, probe=%s", f"{n:,}", _probe is not None)
    return _model, _probe, _device


app = FastAPI(title="LeBall Pong — JEPA", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "probe": _probe is not None}


@app.websocket("/ws-pong")
async def pong_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Pong client connected")

    try:
        model, probe, device = get_model()
    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    # Renderer for encoding frames
    renderer = PongWorld()
    renderer.reset(seed=42)

    # Latent buffer for sliding window
    emb_buffer = []
    action_buffer = []
    prev_ghost = None
    prev_actual = None
    prev_paddles = None
    accuracies = []
    baseline_accuracies = []
    frame_count = 0

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            bx = msg.get("ball_x", 0.5)
            by = msg.get("ball_y", 0.3)
            bvx = msg.get("ball_vx", 0)
            bvy = msg.get("ball_vy", 0)
            pl = msg.get("pad_l", 0.3)
            pr = msg.get("pad_r", 0.3)

            # Detect ball reset (score event) — clear stale context
            if prev_actual is not None:
                jump = abs(bx - prev_actual[0]) + abs(by - prev_actual[1])
                if jump > 0.3:
                    emb_buffer.clear()
                    action_buffer.clear()
                    logger.info("Ball reset detected, cleared context buffers")

            # Set renderer state to match client
            renderer.ball_x = bx
            renderer.ball_y = by
            renderer.ball_vx = bvx
            renderer.ball_vy = bvy
            renderer.paddle_l = pl
            renderer.paddle_r = pr

            # Render and encode current frame
            frame = renderer.render(128)
            frame_t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0

            with torch.no_grad():
                emb = model.encode(frame_t.to(device))[0, 0]
                emb_buffer.append(emb)

                # Zero actions — model trained with normalized actions but we don't
                # have matched scaling. Zero is safer than mismatched values.
                action = torch.zeros(1, 2, device=device)
                action_emb = model.action_encoder(action)[0]
                action_buffer.append(action_emb)

                if len(emb_buffer) > HISTORY_SIZE + 5:
                    emb_buffer = emb_buffer[-(HISTORY_SIZE + 3):]
                if len(action_buffer) > HISTORY_SIZE + 5:
                    action_buffer = action_buffer[-(HISTORY_SIZE + 3):]

                ghost_x, ghost_y = bx, by  # default

                if len(emb_buffer) >= HISTORY_SIZE and probe is not None:
                    # JEPA 1-step prediction
                    ctx = torch.stack(emb_buffer[-HISTORY_SIZE:]).unsqueeze(0)
                    ctx_a = torch.stack(action_buffer[-HISTORY_SIZE:]).unsqueeze(0)
                    pred = model.predict_next(ctx, ctx_a).unsqueeze(0)

                    # Probe extracts state
                    state_norm = probe(pred[0])
                    state = (state_norm[0] * _state_std + _state_mean).cpu().numpy()
                    ghost_x = float(np.clip(state[0], 0, 1))
                    # Probe outputs ball_y normalized to [0,1] but client uses court height CH=0.6
                    ghost_y = float(np.clip(state[1] * 0.6, 0, 0.6))

            # Accuracy: compare previous prediction to current actual
            # Also compute baseline: "predict same position" (no model needed)
            accuracy = None
            baseline = None
            if prev_ghost is not None and prev_actual is not None:
                jepa_dist = np.sqrt((prev_ghost[0] - bx)**2 + (prev_ghost[1] - by)**2)
                # Baseline: predict ball stays where it was last observed
                base_dist = np.sqrt((prev_actual[0] - bx)**2 + (prev_actual[1] - by)**2)

                jepa_acc = max(0, (1 - jepa_dist / 0.3) * 100)
                base_acc = max(0, (1 - base_dist / 0.3) * 100)

                accuracies.append(jepa_acc)
                baseline_accuracies.append(base_acc)
                if len(accuracies) > 50:
                    accuracies = accuracies[-50:]
                if len(baseline_accuracies) > 50:
                    baseline_accuracies = baseline_accuracies[-50:]
                accuracy = np.mean(accuracies)
                baseline = np.mean(baseline_accuracies)

            prev_ghost = (ghost_x, ghost_y)
            prev_actual = (bx, by)

            await ws.send_text(json.dumps({
                "ghost_x": round(ghost_x, 4),
                "ghost_y": round(ghost_y, 4),
                "accuracy": accuracy,
                "baseline": baseline,
            }))

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info("Pong: %d frames, jepa=%.1f%% baseline=%.1f%%",
                            frame_count, accuracy or 0, baseline or 0)

    except WebSocketDisconnect:
        logger.info("Pong disconnected after %d frames", frame_count)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)


def main():
    global _checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    _checkpoint_path = args.checkpoint

    import uvicorn
    logger.info("Starting Pong JEPA on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
