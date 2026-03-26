"""AURA Bounce — Proper JEPA world model inference.

JEPA predicts latent dynamics → state probe extracts ball physics →
BounceWorld renders clean ASCII frames.

Flow:
  1. Client screams → sets initial audio context
  2. Server encodes seed frames from real physics (the launch)
  3. JEPA rolls out latent trajectory (sliding window of 3)
  4. State probe decodes each latent → [ball_x, ball_y, vel_x, vel_y, gravity]
  5. BounceWorld renders ASCII from predicted state
  6. Streams frames to client until ball settles
"""
import argparse
import json
import logging
import time
import asyncio

import numpy as np
import torch
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
        probe = StateProbe(embed_dim=embed_dim)
        probe.load_state_dict(ckpt["probe"])
        probe = probe.to(device).eval()
        _probe = probe
        _state_mean = ckpt["state_mean"].to(device)
        _state_std = ckpt["state_std"].to(device)
        logger.info("State probe loaded (mean=%s, std=%s)", _state_mean.tolist(), _state_std.tolist())
    else:
        logger.warning("No state probe in checkpoint — will use raw physics for rendering")
        _probe = None

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded: %s params, embed_dim=%d", f"{n_params:,}", embed_dim)
    return _model, _probe, _device


app = FastAPI(title="AURA Bounce — Proper JEPA", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "probe_loaded": _probe is not None}


def encode_frames(model, frames_list, device):
    """Encode a list of (H,W) index tensors into JEPA embeddings."""
    stacked = torch.stack(frames_list).unsqueeze(0).to(device)  # (1, T, H, W)
    with torch.no_grad():
        return model.encode(stacked)  # (1, T, D)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    try:
        model, probe, device = get_model()
    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    renderer = BounceWorld()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mode = msg.get("mode", "listen")

            if mode == "listen":
                # Idle: show ball at rest
                renderer.reset(seed=42)
                renderer.ball_x = renderer.W / 2
                renderer.ball_y = renderer.H - 4
                audio_idle = np.zeros(16, dtype=np.float32)
                audio_idle[12] = audio_idle[13] = msg.get("volume", 0.0)
                frame = renderer.render_ascii(audio_idle)
                await ws.send_text(json.dumps({"type": "idle", "frame": frame}))

            elif mode == "launch":
                volume = float(msg.get("volume", 0.5))
                logger.info("LAUNCH! vol=%.2f", volume)

                # --- Generate seed frames with REAL physics ---
                renderer.reset(seed=int(time.time()) % 10000)
                renderer.ball_x = renderer.W / 2
                renderer.ball_y = renderer.H - 4

                launch_audio = np.zeros(16, dtype=np.float32)
                launch_audio[12] = launch_audio[13] = volume
                launch_audio[6] = launch_audio[7] = min(1.0, volume * 1.5)
                launch_audio[0] = launch_audio[1] = volume * 0.8
                zero_audio = np.zeros(16, dtype=np.float32)

                seed_frames = []
                seed_audios = []
                n_seed = HISTORY_SIZE + 1  # 4 frames

                for i in range(n_seed):
                    audio = launch_audio if i == 0 else zero_audio
                    renderer.step(audio)
                    frame_str = renderer.render_ascii(audio)
                    idx = frame_to_indices(frame_str)
                    if not isinstance(idx, torch.Tensor):
                        idx = torch.from_numpy(idx)
                    seed_frames.append(idx.long())
                    seed_audios.append(torch.tensor(audio))

                    # Send seed frames (from real physics)
                    await ws.send_text(json.dumps({
                        "type": "frame", "idx": i,
                        "frame": frame_str, "source": "physics"
                    }))
                    await asyncio.sleep(0.08)

                # --- JEPA latent rollout ---
                with torch.no_grad():
                    # Encode seed frames
                    seed_tensor = torch.stack(seed_frames).unsqueeze(0).to(device)
                    seed_emb = model.encode(seed_tensor)  # (1, 4, D)

                    seed_audio_tensor = torch.stack(seed_audios).unsqueeze(0).to(device)
                    audio_flat = seed_audio_tensor.reshape(-1, 16)
                    seed_audio_emb = model.audio_encoder(audio_flat).reshape(1, -1, EMBED_DIM)

                    emb_list = list(seed_emb[0].unbind(0))  # list of (D,)
                    audio_emb_list = list(seed_audio_emb[0].unbind(0))

                    # Roll out with zero audio (JEPA imagines physics)
                    zero_a = torch.zeros(16, device=device)
                    max_frames = 120

                    for frame_i in range(max_frames):
                        # Sliding window: last 3 embeddings
                        ctx = torch.stack(emb_list[-HISTORY_SIZE:]).unsqueeze(0)
                        ctx_a = torch.stack(audio_emb_list[-HISTORY_SIZE:]).unsqueeze(0)

                        pred = model.predict_next(ctx, ctx_a)  # (1, D)
                        emb_list.append(pred[0])

                        # Encode zero audio for next step
                        zero_a_emb = model.audio_encoder(zero_a.unsqueeze(0))[0]
                        audio_emb_list.append(zero_a_emb)

                        # Decode state with probe
                        if probe is not None:
                            state_norm = probe(pred)  # (1, 5)
                            state = state_norm[0] * _state_std + _state_mean
                            state = state.cpu().numpy()

                            renderer.ball_x = float(np.clip(state[0] * renderer.W, 2, renderer.W - 3))
                            renderer.ball_y = float(np.clip(state[1] * renderer.H, 2, renderer.H - 4))
                            renderer.vel_x = float(state[2] * 10)
                            renderer.vel_y = float(state[3] * 10)
                            renderer.gravity = float(state[4] * 2)

                        # Update trail
                        renderer.trail.append((renderer.ball_x, renderer.ball_y))
                        if len(renderer.trail) > 15:
                            renderer.trail = renderer.trail[-15:]
                        renderer.particles = []

                        # Render and send
                        frame_text = renderer.render_ascii(None)
                        await ws.send_text(json.dumps({
                            "type": "frame", "idx": n_seed + frame_i,
                            "frame": frame_text, "source": "jepa"
                        }))
                        await asyncio.sleep(0.08)

                        # Check settle
                        if renderer.ball_y > renderer.H * 0.85 and abs(renderer.vel_y) < 0.5:
                            break

                        # Check for new launch
                        try:
                            raw2 = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                            msg2 = json.loads(raw2)
                            if msg2.get("mode") == "launch":
                                break
                        except (asyncio.TimeoutError, Exception):
                            pass

                await ws.send_text(json.dumps({"type": "done"}))
                logger.info("Done: %d JEPA frames", frame_i + 1)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
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
    logger.info("Starting AURA Bounce (Proper JEPA) on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
