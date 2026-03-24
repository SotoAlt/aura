"""AURA ASCII — FastAPI inference server for ASCII world models.

Bridges the ASCII world model (JEPA or CNN) to a browser client via WebSocket.
The client sends 16-float audio context vectors; the server returns predicted
ASCII frames as plain text.

Usage:
    # JEPA model
    python -m world_model.infer_ascii \
        --model jepa --checkpoint checkpoints/ascii_jepa.pt --port 8766

    # CNN model
    python -m world_model.infer_ascii \
        --model cnn --checkpoint checkpoints/ascii_cnn.pt --port 8766
"""

import argparse
import json
import logging
import time

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world_model.ascii_model.model import (
    AsciiFramePredictor,
    FRAME_H,
    FRAME_W,
    VOCAB_SIZE,
    indices_to_frame,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.infer_ascii")

# ---------------------------------------------------------------------------
# Globals (singleton model, set on first connection)
# ---------------------------------------------------------------------------

_model = None
_device = None
_model_type = None  # "jepa" or "cnn"
_checkpoint_path = None
_args = None

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------


def get_model():
    """Load the ASCII model on first call; return cached instance thereafter."""
    global _model, _device, _model_type

    if _model is not None:
        return _model, _device, _model_type

    device = _resolve_device(_args.device if _args else "auto")
    _device = device
    _model_type = _args.model if _args else "cnn"

    logger.info("Loading %s model from %s ...", _model_type, _checkpoint_path)
    ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)

    if _model_type == "jepa":
        from world_model.ascii_model.jepa_model import AsciiJEPA

        model = AsciiJEPA()
        model.load_state_dict(ckpt["model"])
    else:
        # Auto-detect hidden size from checkpoint
        conv1_shape = ckpt["model"].get("conv1.weight", None)
        hidden = conv1_shape.shape[0] if conv1_shape is not None else 128
        model = AsciiFramePredictor(hidden=hidden)
        model.load_state_dict(ckpt["model"])

    model = model.to(device).eval()
    _model = model

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded — type=%s  device=%s  params=%s", _model_type, device, f"{n_params:,}")
    return _model, _device, _model_type


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AURA ASCII Inference Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_type": _model_type,
        "checkpoint": str(_checkpoint_path),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """ASCII inference loop.

    Client sends JSON: {"audio": [16 floats]}
    Server returns: plain text ASCII frame (newline-separated rows).
    """
    await ws.accept()
    logger.info("Client connected: %s", ws.client)

    try:
        model, device, model_type = get_model()
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        await ws.send_json({"error": f"Model load failed: {e}"})
        await ws.close(code=1011)
        return

    # Rolling buffer of glyph-index frames: each (1, FRAME_H, FRAME_W) long
    ctx_len = 3 if model_type == "jepa" else 2

    # Seed with real frames from training data if available
    seed_frames = None
    for seed_path in ["data/ascii_wm_v1.npz", "data/ascii_training_v2.npz"]:
        try:
            import numpy as np
            sd = np.load(seed_path)
            seed_frames = torch.from_numpy(sd["frames"][:ctx_len]).long().to(device)
            logger.info("Seeded buffer with %d frames from %s", ctx_len, seed_path)
            break
        except Exception:
            continue

    if seed_frames is not None:
        frame_buffer = [seed_frames[i:i+1] for i in range(ctx_len)]
    else:
        blank = torch.zeros(1, FRAME_H, FRAME_W, dtype=torch.long, device=device)
        frame_buffer = [blank.clone() for _ in range(ctx_len)]

    frame_count = 0
    t_start = time.monotonic()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            audio_list = msg.get("audio")
            if audio_list is None or len(audio_list) != 16:
                await ws.send_json({"error": 'Expected "audio" array of 16 floats'})
                continue

            audio = torch.tensor([audio_list], dtype=torch.float32, device=device)  # (1, 16)

            # Action: [forward, turn] from client WASD
            action_list = msg.get("action")
            if action_list and len(action_list) == 2:
                action = torch.tensor([action_list], dtype=torch.float32, device=device)
            else:
                action = torch.zeros(1, 2, device=device)

            with torch.no_grad():
                if model_type == "jepa":
                    # Encode last 3 frames -> predict latent -> decode
                    ctx_latents = torch.stack(
                        [model.encoder(frame_buffer[-3 + i]) for i in range(3)],
                        dim=1,
                    )  # (1, 3, D)
                    pred_latent = model.predictor(ctx_latents, audio, action)  # (1, D)
                    logits = model.decoder(pred_latent, audio)  # (1, V, 40, 80)
                else:
                    # Stack last 2 frames -> forward
                    prev = torch.cat(frame_buffer[-2:], dim=0).unsqueeze(0)  # (1, 2, H, W)
                    logits = model(prev, audio)  # (1, V, H, W)

            # Temperature sampling to prevent AR collapse
            temperature = 0.7
            probs = torch.softmax(logits / temperature, dim=1)  # (1, V, H, W)
            # Sample from distribution instead of argmax
            B, V, H, W = probs.shape
            pred_indices = torch.multinomial(
                probs.permute(0, 2, 3, 1).reshape(-1, V), 1
            ).reshape(B, H, W)  # (1, H, W)
            frame_text = indices_to_frame(pred_indices[0].cpu())

            # Update rolling buffer
            frame_buffer.append(pred_indices)
            if len(frame_buffer) > ctx_len + 4:
                frame_buffer = frame_buffer[-(ctx_len + 2):]

            await ws.send_text(frame_text)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info("Served %d frames (%.1f fps avg)", frame_count, fps)

    except WebSocketDisconnect:
        elapsed = time.monotonic() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info("Client disconnected after %d frames (%.1f fps avg)", frame_count, fps)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    global _checkpoint_path, _args

    parser = argparse.ArgumentParser(
        description="AURA ASCII inference server — WebSocket bridge"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--model", type=str, default="cnn", choices=["jepa", "cnn"],
                        help="Model type (default: cnn)")
    parser.add_argument("--port", type=int, default=8766,
                        help="Server port (default: 8766)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Inference device (default: auto)")

    args = parser.parse_args()
    _args = args
    _checkpoint_path = args.checkpoint

    from pathlib import Path
    if not Path(_checkpoint_path).exists():
        logger.error("Checkpoint not found: %s", _checkpoint_path)
        raise SystemExit(1)

    import uvicorn
    logger.info("Starting AURA ASCII server on %s:%d (model=%s)", args.host, args.port, args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
