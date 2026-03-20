"""Frame-stack dataset adapter for DIAMOND.

Loads NPZ episodes (from existing data pipeline) and produces:
  - context_frames: (context_frames*3, H, W) — last N RGB frames stacked
  - target_frame: (3, H, W) — next frame to predict
  - audio_context: (16,) — audio conditioning vector

Images are normalized to [-1, 1] (DIAMOND convention).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiamondFrameDataset(Dataset):
    """PyTorch dataset that loads NPZ episodes and serves frame stacks.

    Each sample picks a random episode and random timestep t >= context_frames,
    returning the preceding frames as context and frame[t] as target.
    """

    def __init__(self, data_dir: str, context_frames: int = 4,
                 samples_per_epoch: int = 10000):
        self.data_dir = Path(data_dir)
        self.context_frames = context_frames
        self.samples_per_epoch = samples_per_epoch

        self.episode_paths = sorted(self.data_dir.glob('episode_*.npz'))
        if not self.episode_paths:
            raise FileNotFoundError(f'No episodes found in {data_dir}')

        # Precompute valid (episode_idx, max_t) pairs
        self._episode_lengths = []
        for p in self.episode_paths:
            with np.load(p) as data:
                n_images = len(data['image'])
            self._episode_lengths.append(n_images)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_episode(self, idx: int) -> dict[str, np.ndarray]:
        with np.load(self.episode_paths[idx]) as data:
            return {k: data[k] for k in data.files}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Random episode and timestep (index is ignored — fully random)
        rng = np.random.default_rng(index + np.random.randint(0, 2**31))
        ep_idx = rng.integers(len(self.episode_paths))
        ep = self._load_episode(ep_idx)

        images = ep['image']  # (T+1, H, W, 3) uint8
        contexts = ep['context']  # (T+1, 16) or (T, 16) float32
        n_images = len(images)

        # Need at least context_frames + 1 frames
        min_t = self.context_frames
        max_t = n_images - 1
        if max_t < min_t:
            # Episode too short — pad by repeating first frame
            min_t = max_t

        t = rng.integers(min_t, max_t + 1)

        # Context frames: [t-context_frames, ..., t-1]
        ctx_start = max(0, t - self.context_frames)
        ctx_imgs = images[ctx_start:t]  # (N, H, W, 3)

        # Pad if not enough context (episode start)
        if len(ctx_imgs) < self.context_frames:
            pad_count = self.context_frames - len(ctx_imgs)
            pad = np.repeat(ctx_imgs[:1], pad_count, axis=0)
            ctx_imgs = np.concatenate([pad, ctx_imgs], axis=0)

        # Target frame
        target = images[t]  # (H, W, 3)

        # Audio context at target timestep
        ctx_idx = min(t, len(contexts) - 1)
        audio = contexts[ctx_idx]  # (16,)

        # Normalize images to [-1, 1]
        ctx_imgs = ctx_imgs.astype(np.float32) / 127.5 - 1.0  # (N, H, W, 3)
        target = target.astype(np.float32) / 127.5 - 1.0      # (H, W, 3)

        # Rearrange to channels-first and stack context
        # ctx_imgs: (N, H, W, 3) → (N*3, H, W)
        ctx_imgs = np.transpose(ctx_imgs, (0, 3, 1, 2))  # (N, 3, H, W)
        ctx_imgs = ctx_imgs.reshape(-1, *ctx_imgs.shape[2:])  # (N*3, H, W)
        target = np.transpose(target, (2, 0, 1))  # (3, H, W)

        return {
            'context_frames': torch.from_numpy(ctx_imgs),
            'target_frame': torch.from_numpy(target),
            'audio_context': torch.from_numpy(audio),
        }


def make_dataloader(data_dir: str, cfg: dict,
                    num_workers: int = 2) -> DataLoader:
    """Create a DataLoader for DIAMOND training.

    Args:
        data_dir: Path to NPZ episodes.
        cfg: Config dict with context_frames, batch_size.
        num_workers: DataLoader workers.

    Returns:
        DataLoader yielding batches of frame stacks.
    """
    dataset = DiamondFrameDataset(
        data_dir=data_dir,
        context_frames=cfg['context_frames'],
        samples_per_epoch=max(10000, cfg['batch_size'] * 500),
    )
    return DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
