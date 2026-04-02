"""DIAMOND audio-reactive corridor inference server.

Generates 64×64 corridor frames conditioned on real-time audio features.
Client sends audio FFT data, server returns generated frames as base64 PNG.

Usage:
    python -m world_model.infer_diamond --checkpoint checkpoints/diamond_v2.ckpt --port 8770
"""
import argparse
import base64
import io
import json
import logging
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.diamond")

_model = None
_device = None
_checkpoint_path = None
_context_buffer = []


def get_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    import yaml
    from world_model.diamond.diffusion import EDMDiffusion

    device = torch.device("cpu")
    _device = device
    logger.info("Loading %s", _checkpoint_path)
    ckpt = torch.load(_checkpoint_path, map_location=device, weights_only=False)

    model = EDMDiffusion(ckpt['cfg'])
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    _model = model

    n = sum(p.numel() for p in model.parameters())
    cfg = ckpt['cfg']
    logger.info("Loaded: %s params, %dx%d, %d denoising steps",
                f"{n:,}", cfg['image_size'], cfg['image_size'],
                cfg.get('denoising_steps', 3))
    return _model, _device


app = FastAPI(title="AURA DIAMOND — Audio-Reactive Corridor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.websocket("/ws-diamond")
async def diamond_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("DIAMOND client connected")

    try:
        model, device = get_model()
    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    cfg = model.cfg
    res = cfg['image_size']
    ctx_frames = cfg.get('context_frames', 4)

    # Initialize context with dark frames
    context = [torch.zeros(1, 3, res, res, device=device)] * ctx_frames
    frame_count = 0
    gen_times = []

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # Extract 16-float audio context from client
            audio = torch.zeros(1, 16, device=device)
            if "audio" in msg:
                a = msg["audio"][:16]
                audio[0, :len(a)] = torch.tensor(a, dtype=torch.float32)

            # Generate next frame
            t0 = time.time()
            ctx = torch.cat(context[-ctx_frames:], dim=1)

            with torch.no_grad():
                raw_frame = model.sample(ctx, audio)

            gen_time = time.time() - t0
            gen_times.append(gen_time)
            if len(gen_times) > 30:
                gen_times = gen_times[-30:]

            # Update context
            context.append(raw_frame)
            if len(context) > ctx_frames + 5:
                context = context[-(ctx_frames + 3):]

            # Convert to PNG base64
            img_np = ((raw_frame[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
            img_np = img_np.clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()

            avg_fps = 1.0 / (sum(gen_times) / len(gen_times)) if gen_times else 0

            await ws.send_text(json.dumps({
                "frame": b64,
                "gen_ms": round(gen_time * 1000),
                "fps": round(avg_fps, 1),
                "frame_count": frame_count,
            }))

            frame_count += 1
            if frame_count % 30 == 0:
                logger.info("DIAMOND: %d frames, %.0fms/frame, %.1f fps",
                            frame_count, gen_time * 1000, avg_fps)

    except WebSocketDisconnect:
        logger.info("DIAMOND disconnected after %d frames", frame_count)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)


def main():
    global _checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    _checkpoint_path = args.checkpoint

    import uvicorn
    logger.info("Starting DIAMOND on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
