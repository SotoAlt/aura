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

    def __init__(self, jsonl_path: str | Path):
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._load(jsonl_path)

    def _load(self, jsonl_path: str | Path):
        # Group frames by episode, ordered by step
        episodes: dict[int, list[dict]] = {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ep = rec["episode"]
                if ep not in episodes:
                    episodes[ep] = []
                episodes[ep].append(rec)

        # Sort each episode by step
        for ep in episodes:
            episodes[ep].sort(key=lambda r: r["step"])

        # Build samples: for each triple of consecutive frames in an episode
        for ep_frames in episodes.values():
            # Pre-convert all frames to index grids
            grids = []
            audios = []
            for rec in ep_frames:
                grids.append(frame_to_indices(rec["ascii_frame"]))
                audios.append(torch.tensor(rec["audio_context"], dtype=torch.float32))

            # Generate triplets (prev1, prev2, target)
            for i in range(2, len(grids)):
                self.samples.append((
                    grids[i - 2],       # prev_frame_1
                    grids[i - 1],       # prev_frame_2
                    audios[i],          # audio context for target frame
                    grids[i],           # target frame
                ))

        print(f"Loaded {len(self.samples)} samples from {len(episodes)} episodes")

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
