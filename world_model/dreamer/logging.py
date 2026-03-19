"""Optional wandb logging helpers for AURA training.

Works without wandb installed — all functions are no-ops if wandb is missing.
"""

import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


def init_wandb(cfg: dict, run_name: str = 'aura-train'):
    """Initialize a wandb run. No-op if wandb not installed."""
    if not HAS_WANDB:
        print('[logging] wandb not installed — logging disabled')
        return None
    return wandb.init(project='aura', name=run_name, config=cfg)


def log_metrics(metrics: dict, step: int):
    """Log scalar metrics to wandb. No-op if wandb not active."""
    if not HAS_WANDB or wandb.run is None:
        return
    wandb.log(metrics, step=step)


def log_frames(frames: np.ndarray, step: int, caption: str = 'predicted'):
    """Log a grid of frames to wandb as images.

    Args:
        frames: (N, 64, 64, 3) float32 in [0, 1] or uint8.
        step: Training step.
        caption: Image caption.
    """
    if not HAS_WANDB or wandb.run is None:
        return
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    images = [wandb.Image(f, caption=f'{caption}_{i}') for i, f in enumerate(frames[:8])]
    wandb.log({caption: images}, step=step)


def finish_wandb():
    """Finish the wandb run. No-op if not active."""
    if HAS_WANDB and wandb.run is not None:
        wandb.finish()
