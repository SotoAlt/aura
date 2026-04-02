"""Pong inference with LeWM ViT-Tiny checkpoint.

Same WebSocket API as infer_pong.py but uses the paper's JEPA architecture.
"""
import argparse
import json
import logging
import asyncio
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add le-wm repo to path
sys.path.insert(0, '/tmp/lewm-repo')
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from stable_pretraining.backbone.utils import vit_hf

from world_model.envs.pong_world import PongWorld

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aura.pong.lewm")

HISTORY_SIZE = 3
_model = None
_probe = None
_device = None


def get_model(checkpoint_path):
    global _model, _device
    if _model is not None:
        return _model

    device = torch.device("cpu")
    _device = device
    logger.info("Loading LeWM %s", checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}

    encoder = vit_hf(size='tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    predictor = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
                            depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1)
    action_encoder = Embedder(input_dim=10, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048)
    pred_proj = MLP(input_dim=192, output_dim=192, hidden_dim=2048)

    model = JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    _model = model

    n = sum(p.numel() for p in model.parameters())
    logger.info("Loaded LeWM: %s params", f"{n:,}")
    return model


app = FastAPI(title="LeBall Pong — LeWM ViT-Tiny", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model": "lewm-vit-tiny", "loaded": _model is not None}


@app.websocket("/ws-pong")
async def pong_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Pong LeWM client connected")

    model = get_model(app.state.checkpoint_path)
    renderer = PongWorld()
    renderer.reset(seed=42)

    emb_buffer = []
    action_buffer = []
    prev_ghost = None
    prev_actual = None
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

            # Detect ball reset
            if prev_actual is not None:
                jump = abs(bx - prev_actual[0]) + abs(by - prev_actual[1])
                if jump > 0.3:
                    emb_buffer.clear()
                    action_buffer.clear()

            # Set renderer state and render 224x224
            renderer.ball_x = bx
            renderer.ball_y = by
            renderer.ball_vx = bvx
            renderer.ball_vy = bvy
            renderer.paddle_l = pl
            renderer.paddle_r = pr

            frame = renderer.render(224)
            frame_t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            # Action: pad to 10-dim (frameskip=5 × action_dim=2)
            action = torch.zeros(1, 10)

            with torch.no_grad():
                # Encode single frame
                info = model.encode({'pixels': frame_t.unsqueeze(0), 'action': action.unsqueeze(0)})
                emb = info['emb'][0, 0]  # (192,)
                act_emb = info['act_emb'][0, 0]  # (192,)

                emb_buffer.append(emb)
                action_buffer.append(act_emb)

                if len(emb_buffer) > HISTORY_SIZE + 5:
                    emb_buffer = emb_buffer[-(HISTORY_SIZE + 3):]
                if len(action_buffer) > HISTORY_SIZE + 5:
                    action_buffer = action_buffer[-(HISTORY_SIZE + 3):]

                ghost_x, ghost_y = bx, by

                if len(emb_buffer) >= HISTORY_SIZE:
                    ctx = torch.stack(emb_buffer[-HISTORY_SIZE:]).unsqueeze(0)
                    ctx_a = torch.stack(action_buffer[-HISTORY_SIZE:]).unsqueeze(0)
                    pred = model.predict(ctx, ctx_a)
                    # pred[:, -1] is the predicted next embedding
                    # For now, use cosine distance to estimate ghost position
                    # (no probe trained yet — just use embedding similarity as signal)
                    pred_emb = pred[0, -1]

                    # Simple heuristic: compare predicted embedding to a grid of positions
                    # This is a placeholder — proper approach needs a trained probe
                    # For demo: use the prediction to shift the ghost relative to current position
                    # The embedding difference encodes the predicted motion
                    diff = pred_emb - emb_buffer[-1]
                    # Use first 2 dims as proxy for x,y movement (rough)
                    ghost_x = float(np.clip(bx + diff[0].item() * 0.5, 0, 1))
                    ghost_y = float(np.clip(by + diff[1].item() * 0.3, 0, 0.6))

            # Accuracy
            accuracy = None
            baseline = None
            if prev_ghost is not None and prev_actual is not None:
                jepa_dist = np.sqrt((prev_ghost[0] - bx)**2 + (prev_ghost[1] - by)**2)
                base_dist = np.sqrt((prev_actual[0] - bx)**2 + (prev_actual[1] - by)**2)
                jepa_acc = max(0, (1 - jepa_dist / 0.3) * 100)
                base_acc = max(0, (1 - base_dist / 0.3) * 100)
                accuracies.append(jepa_acc)
                baseline_accuracies.append(base_acc)
                if len(accuracies) > 50: accuracies = accuracies[-50:]
                if len(baseline_accuracies) > 50: baseline_accuracies = baseline_accuracies[-50:]
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
                logger.info("LeWM: %d frames, jepa=%.1f%% baseline=%.1f%%",
                            frame_count, accuracy or 0, baseline or 0)

    except WebSocketDisconnect:
        logger.info("Disconnected after %d frames", frame_count)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app.state.checkpoint_path = args.checkpoint

    import uvicorn
    logger.info("Starting Pong LeWM on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
