"""Autoregressive sampling / imagination for the ASCII frame predictor.

Given seed frames and an audio context sequence, predicts N future frames
by feeding each prediction back as input.

Usage:
    python -m world_model.ascii_model.sample \
        --checkpoint checkpoints/ascii_cnn.pt \
        --steps 32 --output dreams.txt

    # Print to stdout (default):
    python -m world_model.ascii_model.sample \
        --checkpoint checkpoints/ascii_cnn.pt --steps 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from world_model.ascii_model.model import (
    AsciiFramePredictor,
    frame_to_indices,
    indices_to_frame,
    FRAME_H,
    FRAME_W,
)
from world_model.diamond.utils import get_device


def load_seed_frames(
    jsonl_path: str, episode: int = 0, start_step: int = 0
) -> tuple[torch.Tensor, torch.Tensor, list[list[float]]]:
    """Load two seed frames and subsequent audio contexts from a JSONL file.

    Returns:
        (frame1_grid, frame2_grid, list_of_audio_contexts)
    """
    frames: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["episode"] == episode and rec["step"] >= start_step:
                frames.append(rec)
            if len(frames) > 100:
                break

    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(frames)}")

    grid1 = frame_to_indices(frames[0]["ascii_frame"])
    grid2 = frame_to_indices(frames[1]["ascii_frame"])
    audios = [f["audio_context"] for f in frames[2:]]

    return grid1, grid2, audios


def sample(
    checkpoint_path: str,
    steps: int = 32,
    seed_data: str | None = None,
    output: str | None = None,
    device_str: str = "auto",
    temperature: float = 0.8,
):
    """Autoregressively generate frames.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        steps: Number of frames to generate.
        seed_data: JSONL file for seed frames (uses first 2 frames of ep 0).
        output: Output file path. None = stdout.
        device_str: Device string.
        temperature: Sampling temperature (1.0 = categorical, <1 = sharper).
    """
    device = get_device(device_str)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AsciiFramePredictor(embed_dim=16, hidden=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('loss', '?'):.4f}")

    # Seed frames
    if seed_data:
        prev1, prev2, seed_audios = load_seed_frames(seed_data)
    else:
        # Default: blank frames
        prev1 = torch.zeros(FRAME_H, FRAME_W, dtype=torch.long)
        prev2 = torch.zeros(FRAME_H, FRAME_W, dtype=torch.long)
        seed_audios = []

    prev1 = prev1.to(device)
    prev2 = prev2.to(device)

    # Generate audio contexts if we don't have enough
    rng = np.random.default_rng(42)
    while len(seed_audios) < steps:
        t = len(seed_audios) / max(steps, 1)
        ctx = np.zeros(16, dtype=np.float32)
        ctx[0] = ctx[1] = 0.5 + 0.3 * np.sin(2 * np.pi * t * 2)
        ctx[4] = ctx[5] = 0.5 + 0.3 * np.sin(2 * np.pi * t * 3)
        ctx[10] = ctx[11] = t
        ctx[12] = ctx[13] = 0.4 + 0.2 * np.sin(2 * np.pi * t * 5)
        seed_audios.append(ctx.tolist())

    # Autoregressive rollout
    generated_frames: list[str] = []

    with torch.no_grad():
        for i in range(steps):
            # Prepare input
            prev_frames = torch.stack([prev1, prev2], dim=0).unsqueeze(0)  # (1, 2, H, W)
            audio = torch.tensor(
                [seed_audios[i]], dtype=torch.float32, device=device
            )  # (1, 16)

            # Forward
            logits = model(prev_frames, audio)  # (1, V, H, W)

            # Sample with temperature
            if temperature <= 0:
                pred = logits.argmax(dim=1).squeeze(0)  # (H, W)
            else:
                B, V, H, W = logits.shape
                probs = torch.softmax(logits.squeeze(0) / temperature, dim=0)  # (V, H, W)
                probs_flat = probs.permute(1, 2, 0).reshape(-1, V)  # (H*W, V)
                sampled = torch.multinomial(probs_flat, 1).squeeze(1)  # (H*W,)
                pred = sampled.reshape(H, W)

            # Convert to frame string
            frame_str = indices_to_frame(pred)
            generated_frames.append(frame_str)

            # Shift context
            prev1 = prev2
            prev2 = pred

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{steps} frames")

    # Output
    separator = "\n" + "=" * 80 + "\n"
    result = separator.join(generated_frames)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Wrote {steps} frames to {output}")
    else:
        print("\n" + "=" * 80)
        for i, frame in enumerate(generated_frames):
            print(f"--- Frame {i + 1} ---")
            print(frame)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ASCII corridor frames autoregressively"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--steps", type=int, default=32,
                        help="Number of frames to generate")
    parser.add_argument("--seed-data", type=str, default=None,
                        help="JSONL file for seed frames (first 2 of ep 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy, 1 = categorical)")
    args = parser.parse_args()

    sample(
        checkpoint_path=args.checkpoint,
        steps=args.steps,
        seed_data=args.seed_data,
        output=args.output,
        device_str=args.device,
        temperature=args.temperature,
    )
