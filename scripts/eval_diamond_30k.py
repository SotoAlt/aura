"""Evaluate DIAMOND 30K checkpoint — generates GIFs for each audio scenario."""
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from world_model.diamond.sample import load_model, imagine

CKPT = "/workspace/aura/checkpoints/diamond.ckpt"
DATA = "/workspace/video_data"
OUT = "/workspace/eval_30k"

model, cfg = load_model(CKPT)
device = next(model.parameters()).device
C = cfg["context_frames"]

eps = sorted(Path(DATA).glob("episode_*.npz"))
ep = np.load(eps[len(eps) // 2])
seed = ep["image"][:C]

scenarios = {
    "high_energy": np.tile(
        [[0.9] * 2 + [0.9] * 2 + [0.5] * 2 + [0.0] * 2 + [0.6] * 2 + [0.5] * 2 + [0.9] * 2 + [0.0] * 2],
        (50, 1),
    ).astype(np.float32),
    "low_energy": np.tile(
        [[0.05] * 2 + [0.05] * 2 + [0.05] * 2 + [0.0] * 2 + [0.3] * 2 + [0.3] * 2 + [0.05] * 2 + [0.0] * 2],
        (50, 1),
    ).astype(np.float32),
    "forward_motion": np.tile(
        [[0.4] * 2 + [0.5] * 2 + [0.3] * 2 + [0.0] * 2 + [0.6] * 2 + [0.5] * 2 + [0.6] * 2 + [0.0] * 2],
        (50, 1),
    ).astype(np.float32),
}

Path(OUT).mkdir(exist_ok=True)
for name, ctx in scenarios.items():
    print(f"{name}...", end=" ", flush=True)
    frames = imagine(model, seed, ctx, 50, device)
    imgs = [Image.fromarray(f).resize((256, 256), Image.NEAREST) for f in frames]
    imgs[0].save(
        f"{OUT}/{name}.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=100,
        loop=0,
    )
    ff = frames.astype(np.float32) / 255.0
    brightness = np.mean(ff)
    flow = np.mean(np.abs(np.diff(ff, axis=0)))
    print(f"brightness={brightness:.3f} flow={flow:.4f}")

print(f"\nDone! GIFs at {OUT}/")
