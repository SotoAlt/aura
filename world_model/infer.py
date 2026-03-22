"""AURA DIAMOND — FastAPI inference server.

Bridges the DIAMOND diffusion world model to a browser client via WebSocket.
The client sends 16-float audio context vectors; the server returns predicted
frames by running the model autoregressively.

Usage:
    # CPU local testing
    JAX_PLATFORM=cpu python -m world_model.infer \
        --checkpoint checkpoints/diamond.ckpt --port 8765

    # GPU inference
    python -m world_model.infer \
        --checkpoint checkpoints/diamond.ckpt --device cuda
"""

import argparse
import asyncio
import io
import json
import logging
import struct
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world_model.diamond.sample import load_model
from world_model.diamond.utils import get_device, quantize_to_uint8, quantize_to_uint8_np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('aura.infer')

# ---------------------------------------------------------------------------
# Globals (singleton model, set on first connection)
# ---------------------------------------------------------------------------

_model = None
_cfg = None
_device = None
_checkpoint_path = None

# CLI args, populated by main()
_args = None

# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------


def get_model():
    """Load the DIAMOND model on first call; return cached instance thereafter."""
    global _model, _cfg, _device

    if _model is not None:
        return _model, _cfg, _device

    logger.info('Loading model from %s ...', _checkpoint_path)
    device = get_device(_args.device if _args else 'auto')
    _device = device

    model, cfg = load_model(_checkpoint_path, device=device)
    _model = model
    _cfg = cfg

    logger.info('Model loaded — device=%s  image_size=%d  context_frames=%d  denoising_steps=%d',
                device, cfg['image_size'], cfg['context_frames'],
                cfg.get('denoising_steps', 3))
    return _model, _cfg, _device


# ---------------------------------------------------------------------------
# Frame encoding helpers
# ---------------------------------------------------------------------------


def encode_frame_raw(frame_uint8: np.ndarray) -> bytes:
    """Encode (H, W, 3) uint8 frame as raw RGB bytes.

    Returns 3*H*W bytes in row-major RGB order.
    """
    return frame_uint8.tobytes()


def encode_frame_jpeg(frame_uint8: np.ndarray, quality: int = 85) -> bytes:
    """Encode (H, W, 3) uint8 frame as JPEG bytes.

    Falls back to raw RGB if Pillow is not available.
    """
    try:
        from PIL import Image
        img = Image.fromarray(frame_uint8, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        return buf.getvalue()
    except ImportError:
        logger.warning('Pillow not installed — sending raw RGB instead of JPEG')
        return encode_frame_raw(frame_uint8)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title='AURA Inference Server', version='0.1.0')

# Allow browser clients from any origin (dev convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
async def health():
    """Health check — also reports whether the model is loaded."""
    return {
        'status': 'ok',
        'model_loaded': _model is not None,
        'checkpoint': str(_checkpoint_path),
    }


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    """Main inference loop.

    Protocol (client -> server):
        JSON text message with:
        {
            "audio": [16 floats],          # audio context vector
            "format": "raw" | "jpeg",      # optional, default "jpeg"
            "seed": [[H*W*3 uint8], ...]   # optional, only on first message
        }

    Protocol (server -> client):
        Binary message: encoded frame (raw RGB or JPEG depending on format).

    On connect, the server initialises a rolling buffer of 4 black frames.
    Each audio message triggers one model.sample() call to predict the next
    frame, which is appended to the buffer and sent back.
    """
    await ws.accept()
    logger.info('Client connected: %s', ws.client)

    # Lazy-load model on first connection
    try:
        model, cfg, device = get_model()
    except Exception as e:
        logger.error('Failed to load model: %s', e)
        await ws.send_json({'error': f'Model load failed: {e}'})
        await ws.close(code=1011)
        return

    C = cfg['context_frames']  # 4
    H = W = cfg['image_size']  # 64
    output_format = 'jpeg'     # default, client can override

    # Rolling buffer: list of (1, 3, H, W) float tensors in [-1, 1]
    # Initialise with black frames (zeros = mid-gray in [-1,1] → use -1 for black)
    black = torch.full((1, 3, H, W), -1.0, device=device)
    frame_buffer = [black.clone() for _ in range(C)]

    frame_count = 0
    t_start = time.monotonic()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ---- Parse audio context ----------------------------------
            audio_list = msg.get('audio')
            if audio_list is None or len(audio_list) != 16:
                await ws.send_json({'error': 'Expected "audio" array of 16 floats'})
                continue

            audio = torch.tensor([audio_list], dtype=torch.float32, device=device)  # (1, 16)

            # ---- Optional: client overrides output format -------------
            fmt = msg.get('format')
            if fmt in ('raw', 'jpeg'):
                output_format = fmt

            # ---- Optional: client provides seed frames ----------------
            seed = msg.get('seed')
            if seed is not None:
                try:
                    seed_frames = _parse_seed_frames(seed, C, H, W, device)
                    frame_buffer = seed_frames
                    logger.info('Seeded buffer with %d client frames', len(seed_frames))
                except Exception as e:
                    logger.warning('Bad seed data: %s', e)

            # ---- Run model inference ----------------------------------
            context = torch.cat(frame_buffer[-C:], dim=1)  # (1, C*3, H, W)

            pred = model.sample(context, audio)  # (1, 3, H, W) in [-1, 1]

            # Quantize to prevent drift (same as sample.py)
            pred_uint8 = quantize_to_uint8_np(pred)           # (1, 3, H, W) uint8
            pred_clean = pred_uint8.float() / 127.5 - 1.0     # back to [-1, 1]

            # Update rolling buffer (keep last C frames)
            frame_buffer.append(pred_clean)
            if len(frame_buffer) > C + 4:  # keep some slack, trim lazily
                frame_buffer = frame_buffer[-(C + 2):]

            # ---- Encode and send frame --------------------------------
            # Convert (1, 3, H, W) → (H, W, 3) uint8 numpy
            frame_np = pred_uint8[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

            if output_format == 'jpeg':
                payload = encode_frame_jpeg(frame_np)
            else:
                payload = encode_frame_raw(frame_np)

            await ws.send_bytes(payload)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info('Served %d frames (%.1f fps avg)', frame_count, fps)

    except WebSocketDisconnect:
        elapsed = time.monotonic() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info('Client disconnected after %d frames (%.1f fps avg)', frame_count, fps)
    except Exception as e:
        logger.error('WebSocket error: %s', e, exc_info=True)
        try:
            await ws.send_json({'error': str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_seed_frames(seed_data, C: int, H: int, W: int,
                       device: torch.device) -> list[torch.Tensor]:
    """Parse seed frames from client JSON into buffer tensors.

    seed_data: list of lists, each inner list is H*W*3 uint8 values (row-major RGB),
               or list of base64-encoded raw RGB strings.

    Returns list of (1, 3, H, W) float tensors in [-1, 1].
    """
    import base64

    frames = []
    for item in seed_data[:C]:
        if isinstance(item, str):
            # base64-encoded raw RGB
            raw = base64.b64decode(item)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
        elif isinstance(item, list):
            arr = np.array(item, dtype=np.uint8).reshape(H, W, 3)
        else:
            raise ValueError(f'Unexpected seed frame type: {type(item)}')

        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
        frames.append(t)

    # Pad to C frames if fewer provided
    while len(frames) < C:
        frames.insert(0, frames[0].clone())

    return frames


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    global _checkpoint_path, _args

    parser = argparse.ArgumentParser(
        description='AURA DIAMOND inference server — WebSocket bridge')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to DIAMOND .ckpt file')
    parser.add_argument('--port', type=int, default=8765,
                        help='Server port (default: 8765)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Bind address (default: 0.0.0.0)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Inference device (default: auto)')
    parser.add_argument('--format', type=str, default='jpeg',
                        choices=['raw', 'jpeg'],
                        help='Default frame encoding (default: jpeg)')
    parser.add_argument('--preload', action='store_true',
                        help='Load model immediately instead of on first connection')

    args = parser.parse_args()
    _args = args
    _checkpoint_path = args.checkpoint

    if not Path(_checkpoint_path).exists():
        logger.error('Checkpoint not found: %s', _checkpoint_path)
        raise SystemExit(1)

    if args.preload:
        get_model()

    import uvicorn
    logger.info('Starting AURA inference server on %s:%d', args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
