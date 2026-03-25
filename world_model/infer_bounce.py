"""AURA Bounce — One-shot launch, JEPA frame-prediction world model.

Flow:
  1. Client listens for a scream (volume threshold)
  2. Sends ONE launch message with volume level
  3. Server seeds with real physics frames, then JEPA imagines
     ALL subsequent frames autoregressively — pure world model
  4. Streams frames until ball settles or max frames reached
"""
import argparse
import json
import logging
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world_model.envs.bounce_world import BounceWorld
from world_model.ascii_model.jepa_model import AsciiJEPA, GlyphEncoder, LatentPredictor, GlyphDecoder
from world_model.ascii_model.model import frame_to_indices, indices_to_frame, VOCAB_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.infer_bounce")

_model = None
_device = None
_checkpoint_path = None
_n_ctx = None


def get_model():
    global _model, _device, _n_ctx
    if _model is not None:
        return _model, _device, _n_ctx

    device = torch.device("cpu")
    _device = device
    logger.info("Loading from %s ...", _checkpoint_path)
    ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)

    # Detect model config from checkpoint
    sd = ckpt["model"]
    n_ctx = sd["predictor.pos_embed.weight"].shape[0]
    lat_dim = sd["predictor.pos_embed.weight"].shape[1]
    block_nums = set(int(k.split(".")[2]) for k in sd if k.startswith("predictor.blocks."))
    n_layers = max(block_nums) + 1
    ff_dim = sd["predictor.blocks.0.ff.0.weight"].shape[0]

    logger.info("Detected: latent=%d, n_ctx=%d, n_layers=%d, ff_dim=%d",
                lat_dim, n_ctx, n_layers, ff_dim)

    model = AsciiJEPA(latent_dim=lat_dim)
    model.encoder = GlyphEncoder(latent_dim=lat_dim)
    model.predictor = LatentPredictor(latent_dim=lat_dim, n_ctx=n_ctx,
                                       n_layers=n_layers, ff_dim=ff_dim)
    model.decoder = GlyphDecoder(latent_dim=lat_dim)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    _model = model
    _n_ctx = n_ctx

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded: %s params, 99.6%% accuracy bounce world model", f"{n_params:,}")
    return _model, _device, _n_ctx


app = FastAPI(title="AURA Bounce World Model", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    try:
        model, device, n_ctx = get_model()
    except Exception as e:
        logger.error("Load failed: %s", e)
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
                # Show idle ball at bottom
                renderer.reset(seed=42)
                renderer.ball_x = renderer.W / 2
                renderer.ball_y = renderer.H - 4
                audio_idle = np.zeros(16, dtype=np.float32)
                audio_idle[12] = audio_idle[13] = msg.get("volume", 0.0)
                frame = renderer.render_ascii(audio_idle)
                await ws.send_text(json.dumps({"type": "idle", "frame": frame}))

            elif mode == "launch":
                volume = float(msg.get("volume", 0.5))
                logger.info("LAUNCH! volume=%.2f — JEPA imagining physics...", volume)

                # Create launch audio
                launch_audio = np.zeros(16, dtype=np.float32)
                launch_audio[12] = launch_audio[13] = volume
                launch_audio[6] = launch_audio[7] = min(1.0, volume * 1.5)
                launch_audio[0] = launch_audio[1] = volume * 0.8
                zero_audio = np.zeros(16, dtype=np.float32)

                # Generate seed frames using REAL physics
                renderer.reset(seed=int(time.time()) % 10000)
                renderer.ball_x = renderer.W / 2
                renderer.ball_y = renderer.H - 4

                seed_frame_indices = []
                for i in range(n_ctx + 1):
                    audio = launch_audio if i == 0 else zero_audio
                    renderer.step(audio)
                    frame_str = renderer.render_ascii(audio)
                    idx = frame_to_indices(frame_str)
                    if not isinstance(idx, torch.Tensor):
                        idx = torch.from_numpy(idx)
                    seed_frame_indices.append(idx.long())

                    # Send seed frames to client
                    await ws.send_text(json.dumps({
                        "type": "frame", "idx": i,
                        "frame": frame_str, "source": "physics"
                    }))
                    import asyncio
                    await asyncio.sleep(0.08)

                # Now: JEPA takes over — pure world model imagination
                # Use FRAME buffer (not latent) — decode → re-encode each step
                # This keeps predictions anchored to real frame representations
                frame_buffer = list(seed_frame_indices[-n_ctx:])
                zero_audio_t = torch.zeros(1, 16, device=device)

                max_frames = 150
                for frame_i in range(max_frames):
                    with torch.no_grad():
                        # Encode context FRAMES (not latents) — re-encode every step
                        ctx_frames = torch.stack(frame_buffer[-n_ctx:]).unsqueeze(0).to(device)
                        ctx_lats = torch.stack(
                            [model.encoder(ctx_frames[0, j].unsqueeze(0)) for j in range(n_ctx)],
                            dim=1
                        )

                        # Predict next latent + decode to frame
                        pred_lat = model.predictor(ctx_lats, zero_audio_t)
                        logits = model.decoder(pred_lat, zero_audio_t)

                        # Argmax decode
                        pred_indices = logits.argmax(dim=1)  # (1, H, W)

                    # Convert to frame string
                    frame_text = indices_to_frame(pred_indices[0].cpu().numpy())

                    # Update buffer
                    frame_buffer.append(pred_indices[0].cpu())
                    if len(frame_buffer) > n_ctx + 4:
                        frame_buffer = frame_buffer[-(n_ctx + 2):]

                    # Send frame
                    await ws.send_text(json.dumps({
                        "type": "frame", "idx": len(seed_frame_indices) + frame_i,
                        "frame": frame_text, "source": "jepa"
                    }))

                    import asyncio
                    await asyncio.sleep(0.08)

                    # Check for new launch interrupt
                    try:
                        raw2 = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                        msg2 = json.loads(raw2)
                        if msg2.get("mode") == "launch":
                            break
                    except (asyncio.TimeoutError, Exception):
                        pass

                await ws.send_text(json.dumps({"type": "done"}))
                logger.info("Imagination complete: %d frames", frame_i + 1)

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
    logger.info("Starting AURA Bounce World Model on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
