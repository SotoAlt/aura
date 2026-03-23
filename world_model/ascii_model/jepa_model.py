"""JEPA (Joint Embedding Predictive Architecture) for ASCII corridor frames.

Adapts the LeWM concept for discrete glyph prediction:
  - GlyphEncoder: CNN embeds a 40x80 glyph grid into a 192-dim latent
  - LatentPredictor: Transformer predicts next latent from 3 previous + audio
  - GlyphDecoder: Transpose-CNN decodes latent back to glyph logits
  - JEPA loss: MSE on latent prediction + CE on decoded frame + variance reg

~3-5M parameters, trains on CPU or GPU.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.ascii_model.model import VOCAB_SIZE

FRAME_H, FRAME_W = 40, 80
CTX_FRAMES = 3
AUDIO_DIM = 16
LATENT_DIM = 192


# ---------------------------------------------------------------------------
# 1. GlyphEncoder — single frame (B,40,80) -> (B,192)
# ---------------------------------------------------------------------------

class GlyphEncoder(nn.Module):
    def __init__(self, embed_dim: int = 16, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.convs = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, latent_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: (B, 40, 80) long -> (B, latent_dim)"""
        x = self.embed(frame)                        # (B, H, W, E)
        x = x.permute(0, 3, 1, 2)                   # (B, E, H, W)
        x = self.convs(x)                            # (B, latent_dim, H', W')
        x = self.pool(x).flatten(1)                  # (B, latent_dim)
        return x


# ---------------------------------------------------------------------------
# 2. LatentPredictor — Transformer over 3 latents + audio -> predicted next
# ---------------------------------------------------------------------------

class LatentPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        audio_dim: int = AUDIO_DIM,
        n_ctx: int = CTX_FRAMES,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_ctx = n_ctx

        # Audio conditioning via FiLM-style: scale + shift per position
        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim * 2),
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Embedding(n_ctx, latent_dim)

        # Transformer encoder with causal masking
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Projection head for predicted latent
        self.head = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        latents: (B, n_ctx, latent_dim)
        audio:   (B, audio_dim)
        returns: (B, latent_dim) predicted next latent
        """
        B, S, D = latents.shape

        # Audio conditioning: FiLM modulation on each latent
        audio_params = self.audio_mlp(audio)         # (B, 2*D)
        scale, shift = audio_params.chunk(2, dim=1)  # each (B, D)
        scale = scale.unsqueeze(1)                   # (B, 1, D)
        shift = shift.unsqueeze(1)                   # (B, 1, D)
        x = latents * (1.0 + scale) + shift

        # Add positional embeddings
        positions = torch.arange(S, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Causal mask: each position attends only to itself and earlier
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)

        x = self.transformer(x, mask=mask, is_causal=True)

        # Take last position as prediction
        pred = self.head(x[:, -1])                   # (B, D)
        return pred


# ---------------------------------------------------------------------------
# 3. GlyphDecoder — latent (B,192) -> logits (B, VOCAB_SIZE, 40, 80)
# ---------------------------------------------------------------------------

class GlyphDecoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.project = nn.Linear(latent_dim, 128 * 5 * 10)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, VOCAB_SIZE, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> (B, VOCAB_SIZE, 40, 80)"""
        x = self.project(z)                          # (B, 128*5*10)
        x = x.view(-1, 128, 5, 10)                  # (B, 128, 5, 10)
        x = self.deconv(x)                           # (B, VOCAB_SIZE, 40, 80)
        return x


# ---------------------------------------------------------------------------
# 4. AsciiJEPA — full model
# ---------------------------------------------------------------------------

class AsciiJEPA(nn.Module):
    """Joint Embedding Predictive Architecture for ASCII corridor frames.

    forward(frames_seq, audio) where:
        frames_seq: (B, 4, 40, 80) long — 3 context + 1 target
        audio:      (B, 16) float — audio conditioning for the target step
    returns:
        predicted_latent: (B, 192)
        target_latent:    (B, 192) — detached (stop-gradient)
        decoded_logits:   (B, VOCAB_SIZE, 40, 80)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = GlyphEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim)
        self.decoder = GlyphDecoder(latent_dim=latent_dim)

    def forward(self, frames_seq: torch.Tensor, audio: torch.Tensor):
        B = frames_seq.shape[0]
        ctx_frames = frames_seq[:, :CTX_FRAMES]      # (B, 3, 40, 80)
        target_frame = frames_seq[:, CTX_FRAMES]      # (B, 40, 80)

        # Encode each context frame
        ctx_latents = []
        for i in range(CTX_FRAMES):
            ctx_latents.append(self.encoder(ctx_frames[:, i]))
        ctx_latents = torch.stack(ctx_latents, dim=1)  # (B, 3, D)

        # Encode target frame (stop-gradient)
        target_latent = self.encoder(target_frame).detach()

        # Predict next latent from context + audio
        predicted_latent = self.predictor(ctx_latents, audio)

        # Decode predicted latent to glyph logits
        decoded_logits = self.decoder(predicted_latent)

        return predicted_latent, target_latent, decoded_logits

    @staticmethod
    def variance_regularization(latents: torch.Tensor, min_std: float = 0.1) -> torch.Tensor:
        """Soft penalty if per-dimension std drops below min_std."""
        std = latents.std(dim=0)                      # (D,)
        penalty = F.relu(min_std - std).pow(2).mean()
        return penalty


# ---------------------------------------------------------------------------
# 5. Training script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train AsciiJEPA")
    pa.add_argument("--data", default="data/ascii_training.npz")
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--checkpoint", default="checkpoints/ascii_jepa.pt")
    pa.add_argument("--device", default="cpu")
    args = pa.parse_args()

    # --- Load data ---
    d = np.load(args.data)
    frames = torch.from_numpy(d["frames"]).long()     # (N, 40, 80)
    audio = torch.from_numpy(d["audios"]).float()      # (N, 16)

    # Build 4-frame windows: frames[i-3:i+1] as (3 ctx + 1 target)
    n_ctx = CTX_FRAMES
    seq = torch.stack([frames[i - n_ctx : i + 1] for i in range(n_ctx, len(frames))])
    aud = audio[n_ctx:]
    tgt = frames[n_ctx:]
    print(f"Dataset: {len(seq)} samples from {len(frames)} frames")

    # --- Device ---
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = AsciiJEPA().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AsciiJEPA: {n_params:,} parameters on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bs = args.batch_size

    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(seq))
        e_pred, e_dec, e_reg, e_correct, e_total = 0.0, 0.0, 0.0, 0, 0

        for i in range(0, len(seq), bs):
            idx = perm[i : i + bs]
            xb = seq[idx].to(device)
            ab = aud[idx].to(device)
            yb = tgt[idx].to(device)

            pred_lat, tgt_lat, logits = model(xb, ab)

            # Losses
            l_pred = F.mse_loss(pred_lat, tgt_lat)
            l_decode = F.cross_entropy(logits, yb)
            l_sigreg = AsciiJEPA.variance_regularization(pred_lat)
            loss = l_pred + l_decode + 0.1 * l_sigreg

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Accuracy
            preds = logits.argmax(dim=1)
            correct = (preds == yb).sum().item()
            total = yb.numel()

            e_pred += l_pred.item() * len(idx)
            e_dec += l_decode.item() * len(idx)
            e_reg += l_sigreg.item() * len(idx)
            e_correct += correct
            e_total += total
            step += 1

            if step % 50 == 0:
                print(
                    f"  step {step:>5d}  "
                    f"L_pred={l_pred.item():.4f}  "
                    f"L_dec={l_decode.item():.4f}  "
                    f"L_reg={l_sigreg.item():.4f}  "
                    f"acc={correct / total * 100:.1f}%"
                )

        n = len(seq)
        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"L_pred={e_pred / n:.4f}  "
            f"L_dec={e_dec / n:.4f}  "
            f"L_reg={e_reg / n:.4f}  "
            f"acc={e_correct / e_total * 100:.1f}%"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "vocab_size": VOCAB_SIZE, "epoch": epoch + 1},
                args.checkpoint,
            )
            print(f"  -> checkpoint saved: {args.checkpoint}")

    # Final save
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "vocab_size": VOCAB_SIZE, "epoch": args.epochs},
        args.checkpoint,
    )
    print(f"Saved final checkpoint -> {args.checkpoint}")
