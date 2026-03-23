"""Training script for the ASCII corridor frame predictor.

Usage:
    python -m world_model.ascii_model.train \
        --data data/ascii_training.jsonl \
        --epochs 50 --batch-size 32 \
        --checkpoint checkpoints/ascii_cnn.pt

Designed to train on CPU (M3 MacBook) in under an hour on 20K frames.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from world_model.ascii_model.model import AsciiFramePredictor, VOCAB_SIZE
from world_model.ascii_model.dataset import make_dataloader
from world_model.diamond.utils import get_device


def train(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_path: str = "checkpoints/ascii_cnn.pt",
    device_str: str = "auto",
    save_every: int = 5,
    log_every: int = 100,
):
    """Main training loop.

    Args:
        data_path: Path to JSONL training data.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        checkpoint_path: Where to save checkpoints.
        device_str: 'auto', 'cpu', 'cuda', or 'mps'.
        save_every: Save checkpoint every N epochs.
        log_every: Log loss every N steps.
    """
    device = get_device(device_str)
    print(f"Device: {device}")

    # Data
    loader = make_dataloader(data_path, batch_size=batch_size, shuffle=True)
    print(f"Batches per epoch: {len(loader)}")

    # Model
    model = AsciiFramePredictor(embed_dim=16, hidden=64).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.2f}M)")
    print(f"Vocabulary: {VOCAB_SIZE} glyphs")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print()

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        step_loss = 0.0
        t_start = time.time()

        for batch_idx, (prev_frames, audio, target) in enumerate(loader):
            prev_frames = prev_frames.to(device)   # (B, 2, H, W) long
            audio = audio.to(device)                # (B, 16) float
            target = target.to(device)              # (B, H, W) long

            # Forward
            logits = model(prev_frames, audio)      # (B, V, H, W)

            # Cross-entropy loss per character position
            B, V, H, W = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * H * W, V),
                target.reshape(B * H * W),
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Stats
            loss_val = loss.item()
            epoch_loss += loss_val
            step_loss += loss_val
            global_step += 1

            # Accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=1)  # (B, H, W)
                epoch_correct += (preds == target).sum().item()
                epoch_total += target.numel()

            # Log every N steps
            if global_step % log_every == 0:
                avg = step_loss / log_every
                elapsed = time.time() - t_start
                steps_sec = log_every / max(elapsed, 1e-6)
                print(
                    f"  step {global_step:>6d}  "
                    f"loss={avg:.4f}  "
                    f"{steps_sec:.1f} steps/s"
                )
                step_loss = 0.0
                t_start = time.time()

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(len(loader), 1)
        accuracy = epoch_correct / max(epoch_total, 1)
        print(
            f"Epoch {epoch:>3d}/{epochs}  "
            f"loss={avg_epoch_loss:.4f}  "
            f"acc={accuracy:.4f}"
        )

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = Path(checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "accuracy": accuracy,
                "vocab_size": VOCAB_SIZE,
            }
            torch.save(ckpt, str(ckpt_path))
            tag = " *best*" if avg_epoch_loss < best_loss else ""
            print(f"  Checkpoint saved: {ckpt_path}{tag}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = ckpt_path.with_stem(ckpt_path.stem + "_best")
                torch.save(ckpt, str(best_path))

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Final checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ASCII corridor frame predictor"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to JSONL training data",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/ascii_cnn.pt",
        help="Checkpoint save path",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda, mps",
    )
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log loss every N steps")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        device_str=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
    )
