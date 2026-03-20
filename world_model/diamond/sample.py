"""Autoregressive imagination for DIAMOND.

The money function: given seed frames and audio contexts, generates
a sequence of future frames by feeding each prediction back as input.

Usage:
    python -m world_model.diamond.sample \
        --checkpoint checkpoints/diamond.ckpt \
        --data data/matsya \
        --output sample_output/ \
        --frames 50
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model.diamond.diffusion import EDMDiffusion
from world_model.diamond.utils import get_device, quantize_to_uint8, quantize_to_uint8_np


def load_model(checkpoint_path: str,
               device: torch.device | None = None,
               use_ema: bool = True) -> tuple[EDMDiffusion, dict]:
    """Load a trained DIAMOND model from checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file.
        device: Target device. Auto-detects if None.
        use_ema: Use EMA weights (recommended for generation).

    Returns:
        (model, cfg) tuple.
    """
    if device is None:
        device = get_device()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    model = EDMDiffusion(cfg).to(device)

    weights_key = 'ema_model' if use_ema and 'ema_model' in ckpt else 'model'
    model.load_state_dict(ckpt[weights_key])
    model.eval()

    return model, cfg


@torch.no_grad()
def imagine(model: EDMDiffusion, seed_frames: np.ndarray,
            audio_contexts: np.ndarray, n_steps: int,
            device: torch.device | None = None) -> np.ndarray:
    """Autoregressively imagine future frames.

    Args:
        model: Trained DIAMOND model (eval mode).
        seed_frames: (S, H, W, 3) uint8 — at least context_frames real frames.
        audio_contexts: (N, 16) float32 — audio context for each generated frame.
            N >= n_steps.
        n_steps: Number of frames to generate.
        device: Target device.

    Returns:
        (n_steps, H, W, 3) uint8 generated frames.
    """
    if device is None:
        device = next(model.parameters()).device

    C = model.cfg['context_frames']  # context window size (4)

    # Initialize buffer with seed frames, normalized to [-1, 1]
    # seed_frames: (S, H, W, 3) uint8 → tensors (1, 3, H, W) float
    buffer = []
    for i in range(len(seed_frames)):
        frame = seed_frames[i].astype(np.float32) / 127.5 - 1.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
        buffer.append(frame)

    # Pad buffer if fewer than C seed frames
    while len(buffer) < C:
        buffer.insert(0, buffer[0])

    generated = []

    for t in range(n_steps):
        # Stack last C frames as context
        context = torch.cat(buffer[-C:], dim=1)  # (1, C*3, H, W)

        # Audio conditioning
        ctx_idx = min(t, len(audio_contexts) - 1)
        audio = torch.from_numpy(audio_contexts[ctx_idx:ctx_idx + 1]).to(device)

        # Generate next frame
        pred = model.sample(context, audio)  # (1, 3, H, W) in [-1, 1]

        # Quantize to uint8 and back (prevent floating-point drift)
        pred_uint8 = quantize_to_uint8_np(pred)
        pred_clean = pred_uint8.float() / 127.5 - 1.0

        # Store
        buffer.append(pred_clean)
        generated.append(pred_uint8[0].permute(1, 2, 0).cpu().numpy())

    return np.stack(generated)  # (n_steps, H, W, 3) uint8


def imagine_from_data(checkpoint_path: str, data_dir: str,
                      n_steps: int = 50, episode_idx: int = 0,
                      seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load checkpoint + data, imagine from a specific episode.

    Args:
        checkpoint_path: Path to checkpoint.
        data_dir: Path to NPZ episodes.
        n_steps: Frames to generate.
        episode_idx: Which episode to seed from.
        seed_offset: Frame offset within episode for seed.

    Returns:
        (generated_frames, audio_contexts) — uint8 frames and contexts used.
    """
    model, cfg = load_model(checkpoint_path)
    device = next(model.parameters()).device
    C = cfg['context_frames']

    # Load episode
    episodes = sorted(Path(data_dir).glob('episode_*.npz'))
    if not episodes:
        raise FileNotFoundError(f'No episodes in {data_dir}')
    ep_idx = min(episode_idx, len(episodes) - 1)
    ep = np.load(episodes[ep_idx])

    images = ep['image']  # (T+1, H, W, 3) uint8
    contexts = ep['context']  # (T+1, 16) or (T, 16)

    # Seed frames
    start = min(seed_offset, len(images) - C - 1)
    seed = images[start:start + C]

    # Audio contexts for generation
    ctx_start = start + C
    audio_ctx = contexts[ctx_start:ctx_start + n_steps]
    # Pad if not enough contexts
    if len(audio_ctx) < n_steps:
        pad = np.repeat(audio_ctx[-1:], n_steps - len(audio_ctx), axis=0)
        audio_ctx = np.concatenate([audio_ctx, pad])

    frames = imagine(model, seed, audio_ctx, n_steps, device)
    return frames, audio_ctx


if __name__ == '__main__':
    from world_model.eval import make_gif

    parser = argparse.ArgumentParser(description='DIAMOND imagination')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='sample_output/')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Imagining {args.frames} frames from episode {args.episode}...')
    frames, audio_ctx = imagine_from_data(
        args.checkpoint, args.data, args.frames,
        args.episode, args.seed_offset,
    )

    gif_path = out / 'imagination.gif'
    make_gif(frames, str(gif_path), fps=10, scale=4)
    print(f'Saved: {gif_path} ({len(frames)} frames)')
