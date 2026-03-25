"""Tiny CNN frame predictor for the ASCII corridor world model.

Predicts the next 40x80 glyph frame given two previous frames and a
16-float audio context vector.  ~200-300K parameters, designed to train
on CPU in under an hour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Glyph vocabulary — sourced from world_model/envs/ascii_corridor.py
# ---------------------------------------------------------------------------

# All glyphs that appear in the corridor renderer palettes:
#   WALL_GLYPHS_COOL: █▓▒░╔═╗║
#   WALL_GLYPHS_WARM: █▓▒░╭─╮│
#   FLOOR_GLYPHS:     ·.:;(space)
#   CEIL_GLYPHS:      `'"*(space)
#   FLASH_CHARS:       ✦✧★☆⚡✹✸
#   INTENSE_WALL:      ▉▊▋▌
# Plus space (appears in multiple palettes) and period/dot.

_ALL_GLYPHS = (
    " "           # 0 — space (most common, also used for fog)
    "█▓▒░"        # wall shading blocks
    "╔═╗║"        # cool box-drawing
    "╭─╮│"        # warm box-drawing
    "·.:;"        # floor
    "`'\"*"       # ceiling
    "✦✧★☆⚡✹✸"    # flash
    "▉▊▋▌"        # intense wall
    "•●◉⬤"        # ball glyphs (bounce world)
    "∙°˚"         # trail glyphs (bounce world)
    "○╚╝"         # ball edge + corners (bounce world)
)

# Deduplicate while preserving order, then add UNKNOWN
_seen: set[str] = set()
GLYPHS: list[str] = []
for ch in _ALL_GLYPHS:
    if ch not in _seen:
        _seen.add(ch)
        GLYPHS.append(ch)
GLYPHS.append("\x00")  # UNKNOWN token (last index)

VOCAB_SIZE = len(GLYPHS)
GLYPH_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GLYPHS)}
IDX_TO_GLYPH: dict[int, str] = {i: g for i, g in enumerate(GLYPHS)}
UNK_IDX = GLYPH_TO_IDX["\x00"]

FRAME_H = 40
FRAME_W = 80


def frame_to_indices(frame_str: str) -> torch.Tensor:
    """Convert an ASCII frame string (newline-separated) to a (H, W) int tensor."""
    lines = frame_str.split("\n")
    grid = torch.full((FRAME_H, FRAME_W), UNK_IDX, dtype=torch.long)
    for r, line in enumerate(lines[:FRAME_H]):
        for c, ch in enumerate(line[:FRAME_W]):
            grid[r, c] = GLYPH_TO_IDX.get(ch, UNK_IDX)
    return grid


def indices_to_frame(grid: torch.Tensor) -> str:
    """Convert a (H, W) index tensor back to an ASCII frame string."""
    lines = []
    for r in range(grid.shape[0]):
        chars = []
        for c in range(grid.shape[1]):
            idx = grid[r, c].item()
            chars.append(IDX_TO_GLYPH.get(idx, " "))
        lines.append("".join(chars))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FiLM conditioning layer
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: audio context -> (scale, shift)."""

    def __init__(self, audio_dim: int, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, channels * 2),
        )

    def forward(self, audio: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, audio_dim)
            feature_map: (B, C, H, W)
        Returns:
            Modulated feature map (B, C, H, W).
        """
        params = self.mlp(audio)  # (B, 2*C)
        scale, shift = params.chunk(2, dim=1)  # each (B, C)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Tiny CNN frame predictor
# ---------------------------------------------------------------------------

class AsciiFramePredictor(nn.Module):
    """Predict next ASCII frame from 2 previous frames + audio context.

    Input:
        prev_frames: (B, 2, H, W) long — glyph indices for 2 previous frames
        audio_context: (B, 16) float — audio feature vector

    Output:
        logits: (B, vocab_size, H, W)
    """

    def __init__(self, embed_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        in_channels = 2 * embed_dim  # 2 frames * embed_dim

        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden, VOCAB_SIZE, 1)

        self.film1 = FiLM(16, hidden)
        self.film2 = FiLM(16, hidden)

    def forward(self, prev_frames: torch.Tensor, audio_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_frames: (B, 2, H, W) long
            audio_context: (B, 16) float

        Returns:
            (B, VOCAB_SIZE, H, W) logits
        """
        B, N, H, W = prev_frames.shape
        # Embed each glyph: (B, 2, H, W) -> (B, 2, H, W, embed_dim)
        x = self.embed(prev_frames)
        # Reshape to (B, 2*embed_dim, H, W)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)

        # Conv block 1
        h = F.relu(self.conv1(x))
        h = self.film1(audio_context, h)

        # Conv block 2 + residual
        h2 = F.relu(self.conv2(h))
        h2 = self.film2(audio_context, h2)
        h = h + h2  # residual

        # Conv block 3 + residual
        h3 = F.relu(self.conv3(h))
        h = h + h3  # residual

        # Output logits
        logits = self.conv4(h)  # (B, VOCAB_SIZE, H, W)
        return logits
