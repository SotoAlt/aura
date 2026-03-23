"""PyTorch Dataset for ASCII corridor frame prediction.

Loads a JSONL file of frames (produced by ascii_corridor.py --generate),
groups them by episode, and yields (prev_frame_1, prev_frame_2, audio_context,
target_frame) tuples.  Episode boundaries are respected — we never use frames
from different episodes as context.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from world_model.ascii_model.model import frame_to_indices, FRAME_H, FRAME_W, UNK_IDX


class AsciiFrameDataset(Dataset):
    """Dataset of (prev1, prev2, audio, target) frame tuples.

    Requires at least 3 consecutive frames per episode (2 context + 1 target).
    """

    def __init__(self, data_path: str | Path):
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        data_path = Path(data_path)
        # Use NPZ cache if available (100x faster), fall back to JSONL
        npz_path = data_path.with_suffix('.npz')
        if npz_path.exists():
            self._load_npz(npz_path)
        else:
            self._load_jsonl(data_path)

    def _load_npz(self, npz_path: Path):
        """Fast path: load pre-converted NPZ cache."""
        import numpy as np
        data = np.load(npz_path)
        all_frames = torch.from_numpy(data['frames'].astype(np.int64))
        all_audios = torch.from_numpy(data['audios'])
        all_episodes = data['episodes']

        # Group by episode and build triplets
        eps = {}
        for i in range(len(all_episodes)):
            ep = int(all_episodes[i])
            if ep not in eps:
                eps[ep] = []
            eps[ep].append(i)

        for indices in eps.values():
            for j in range(2, len(indices)):
                self.samples.append((
                    all_frames[indices[j - 2]],
                    all_frames[indices[j - 1]],
                    all_audios[indices[j]],
                    all_frames[indices[j]],
                ))
        print(f"Loaded {len(self.samples)} samples from {len(eps)} episodes (NPZ)")

    def _load_jsonl(self, jsonl_path: Path):
        """Slow path: parse JSONL directly."""
        episodes: dict[int, list[dict]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ep = rec["episode"]
                if ep not in episodes:
                    episodes[ep] = []
                episodes[ep].append(rec)

        for ep in episodes:
            episodes[ep].sort(key=lambda r: r["step"])

        for ep_frames in episodes.values():
            grids = []
            audios = []
            for rec in ep_frames:
                grids.append(frame_to_indices(rec["ascii_frame"]))
                audios.append(torch.tensor(rec["audio_context"], dtype=torch.float32))

            for i in range(2, len(grids)):
                self.samples.append((
                    grids[i - 2],
                    grids[i - 1],
                    audios[i],
                    grids[i],
                ))
        print(f"Loaded {len(self.samples)} samples from {len(episodes)} episodes (JSONL)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        prev1, prev2, audio, target = self.samples[idx]
        # Stack prev frames: (2, H, W)
        prev_frames = torch.stack([prev1, prev2], dim=0)
        return prev_frames, audio, target


def make_dataloader(
    jsonl_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for ASCII frame training."""
    ds = AsciiFrameDataset(jsonl_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
