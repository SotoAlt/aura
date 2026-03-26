"""Proper JEPA world model following LeWM paper exactly.

Architecture (from paper):
  - Encoder: CNN → Projector MLP (with BatchNorm)
  - Predictor: AdaLN-zero Transformer with causal masking
  - Pred Projector: same MLP architecture
  - SIGReg on ALL embeddings, λ=0.09
  - lr=5e-5, AdamW, gradient clip 1.0
  - Teacher-forced training (NO AR rollouts)
  - Sliding window of 3 during inference

After JEPA training:
  - Train a tiny state probe (linear) to extract [ball_x, ball_y, vel] from latent
  - BounceWorld renders from predicted state

~4M params. Trains in <1 hour on A100.
"""
from __future__ import annotations

import argparse
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.ascii_model.model import VOCAB_SIZE

FRAME_H, FRAME_W = 40, 80
AUDIO_DIM = 16
EMBED_DIM = 192
HISTORY_SIZE = 3  # context window
PROJ_HIDDEN = 1024  # projector MLP hidden (paper uses 2048 but we're smaller)


# ---------------------------------------------------------------------------
# SIGReg — exact copy from LeWM
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D) — ALL embeddings across time and batch."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Projector MLP with BatchNorm (from paper)
# ---------------------------------------------------------------------------

class ProjectorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=PROJ_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """x: (B, D) or (B*T, D)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# AdaLN-zero Transformer Block (from paper)
# ---------------------------------------------------------------------------

class ConditionalBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )
        # AdaLN-zero: 6 modulation params, initialized to zero
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, mask=None):
        """x: (B,T,D), c: (B,T,D) conditioning per position."""
        s1, sh1, g1, s2, sh2, g2 = self.adaLN(c).chunk(6, dim=-1)
        # Attention
        h = self.norm1(x) * (1 + s1) + sh1
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=(mask is not None))
        x = x + g1 * h
        # FFN
        h = self.norm2(x) * (1 + s2) + sh2
        x = x + g2 * self.ff(h)
        return x


# ---------------------------------------------------------------------------
# Encoder: CNN for 40x80 glyph frames
# ---------------------------------------------------------------------------

class GlyphEncoder(nn.Module):
    def __init__(self, embed_dim=16, hidden_dim=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.convs = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frame):
        """frame: (B, H, W) long → (B, hidden_dim)"""
        x = self.embed(frame).permute(0, 3, 1, 2)
        return self.pool(self.convs(x)).flatten(1)


# ---------------------------------------------------------------------------
# Audio Encoder (like LeWM's action encoder)
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    def __init__(self, audio_dim=AUDIO_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, 4 * embed_dim), nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, audio):
        """audio: (B, 16) → (B, embed_dim)"""
        return self.net(audio)


# ---------------------------------------------------------------------------
# ARPredictor — causal transformer with AdaLN-zero
# ---------------------------------------------------------------------------

class ARPredictor(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=8, n_layers=6,
                 ff_dim=1024, dropout=0.1, max_len=HISTORY_SIZE):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            ConditionalBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, cond):
        """x: (B,T,D) frame embeddings, cond: (B,T,D) audio embeddings."""
        T = x.size(1)
        x = x + self.pos_embed[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for block in self.blocks:
            x = block(x, cond, mask=mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full JEPA World Model
# ---------------------------------------------------------------------------

class JEPAWorldModel(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=8, n_layers=6, ff_dim=1024):
        super().__init__()
        hidden_dim = embed_dim

        self.encoder = GlyphEncoder(hidden_dim=hidden_dim)
        self.projector = ProjectorMLP(hidden_dim, embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        self.predictor = ARPredictor(embed_dim, n_heads, n_layers, ff_dim)
        self.pred_projector = ProjectorMLP(embed_dim, embed_dim)
        self.sigreg = SIGReg()

    def encode(self, frames):
        """frames: (B, T, H, W) → (B, T, embed_dim) projected embeddings."""
        B, T = frames.shape[:2]
        flat = frames.reshape(B * T, *frames.shape[2:])
        h = self.encoder(flat)  # (B*T, hidden)
        e = self.projector(h)   # (B*T, embed)
        return e.reshape(B, T, -1)

    def forward(self, frames, audio):
        """
        frames: (B, T, H, W) — T = history_size + 1 (e.g. 4 frames)
        audio: (B, T, 16) — audio for each frame
        Returns: pred_loss, sigreg_loss, pred_emb, tgt_emb
        """
        B, T = frames.shape[:2]

        # Encode all frames
        emb = self.encode(frames)  # (B, T, D)

        # Encode audio for context frames
        audio_flat = audio.reshape(B * T, -1)
        audio_emb = self.audio_encoder(audio_flat).reshape(B, T, -1)

        # Context: first HISTORY_SIZE frames
        ctx_emb = emb[:, :HISTORY_SIZE]      # (B, 3, D)
        ctx_audio = audio_emb[:, :HISTORY_SIZE]  # (B, 3, D)

        # Predict: causal transformer over context
        pred_raw = self.predictor(ctx_emb, ctx_audio)  # (B, 3, D)

        # Project predictions
        pred_proj = self.pred_projector(pred_raw.reshape(B * HISTORY_SIZE, -1))
        pred_emb = pred_proj.reshape(B, HISTORY_SIZE, -1)  # (B, 3, D)

        # Target: shifted by 1 (frames 1,2,3)
        tgt_emb = emb[:, 1:HISTORY_SIZE + 1]  # (B, 3, D)

        # Losses
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))  # (T, B, D) — ALL embeddings

        return pred_loss, sigreg_loss, pred_emb, tgt_emb

    def predict_next(self, ctx_emb, ctx_audio):
        """Single-step prediction for inference.
        ctx_emb: (B, <=3, D), ctx_audio: (B, <=3, D)
        Returns: (B, D) next predicted embedding.
        """
        was_training = self.pred_projector.training
        self.pred_projector.eval()  # BatchNorm needs eval for B=1
        pred_raw = self.predictor(ctx_emb, ctx_audio)
        pred_proj = self.pred_projector(pred_raw[:, -1])
        if was_training:
            self.pred_projector.train()
        return pred_proj

    def rollout(self, seed_frames, seed_audio, future_audio, n_steps):
        """Autoregressive rollout in latent space.
        seed_frames: (B, S, H, W) — S seed frames
        seed_audio: (B, S, 16)
        future_audio: (B, n_steps, 16)
        Returns: (B, n_steps, D) predicted embeddings
        """
        # Encode seeds
        emb = self.encode(seed_frames)  # (B, S, D)
        B = emb.shape[0]

        audio_flat = seed_audio.reshape(B * seed_audio.shape[1], -1)
        audio_emb_list = list(self.audio_encoder(audio_flat).reshape(B, -1, EMBED_DIM).unbind(1))
        emb_list = list(emb.unbind(1))

        predictions = []
        for t in range(n_steps):
            # Sliding window: last HISTORY_SIZE embeddings
            ctx = torch.stack(emb_list[-HISTORY_SIZE:], dim=1)
            ctx_a = torch.stack(audio_emb_list[-HISTORY_SIZE:], dim=1)

            pred = self.predict_next(ctx, ctx_a)  # (B, D)
            predictions.append(pred)
            emb_list.append(pred)

            # Encode future audio
            fa = self.audio_encoder(future_audio[:, t])  # (B, D)
            audio_emb_list.append(fa)

        return torch.stack(predictions, dim=1)  # (B, n_steps, D)


# ---------------------------------------------------------------------------
# State Probe — trained AFTER JEPA, extracts ball state from latent
# ---------------------------------------------------------------------------

class StateProbe(nn.Module):
    """Linear probe: latent → [ball_x, ball_y, vel_x, vel_y, gravity]."""
    def __init__(self, embed_dim=EMBED_DIM, state_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train proper JEPA (LeWM-aligned)")
    pa.add_argument("--data", required=True)
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--checkpoint", default="checkpoints/jepa_proper.pt")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--probe-epochs", type=int, default=30,
                    help="Epochs for state probe training after JEPA")
    args = pa.parse_args()

    # Load data
    d = np.load(args.data)
    frames = torch.from_numpy(d["frames"]).long()
    audio = torch.from_numpy(d["audios"]).float()
    episodes = d.get("episodes", None)
    has_states = "states" in d
    states = torch.from_numpy(d["states"]).float() if has_states else None

    # Build 4-frame windows (history=3 + 1 target), respecting episodes
    windows, win_audio, win_states = [], [], []
    for i in range(HISTORY_SIZE, len(frames)):
        if episodes is not None and episodes[i] != episodes[i - HISTORY_SIZE]:
            continue
        windows.append(frames[i - HISTORY_SIZE: i + 1])  # (4, H, W)
        win_audio.append(audio[i - HISTORY_SIZE: i + 1])  # (4, 16)
        if has_states:
            win_states.append(states[i])

    windows = torch.stack(windows)       # (N, 4, H, W)
    win_audio = torch.stack(win_audio)   # (N, 4, 16)
    if has_states:
        win_states = torch.stack(win_states)
    print(f"Dataset: {len(windows)} windows from {len(frames)} frames")

    # Device
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # Model — paper hyperparams
    model = JEPAWorldModel(
        embed_dim=EMBED_DIM, n_heads=8, n_layers=6, ff_dim=1024,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"JEPAWorldModel: {n_params:,} params on {device}")

    # Optimizer — from paper: AdamW, lr=5e-5, wd=1e-3, grad_clip=1.0
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    SIGREG_LAMBDA = 0.09  # from paper
    bs = args.batch_size

    # =====================================================
    # Phase 1: JEPA training (teacher-forced, no AR)
    # =====================================================
    print(f"\n=== JEPA Training: {args.epochs} epochs, lr=5e-5, λ={SIGREG_LAMBDA} ===")

    for epoch in range(args.epochs):
        perm = torch.randperm(len(windows))
        e_pred, e_reg, n = 0.0, 0.0, 0
        model.train()

        for i in range(0, len(windows), bs):
            idx = perm[i:i + bs]
            xb = windows[idx].to(device)  # (B, 4, H, W)
            ab = win_audio[idx].to(device)  # (B, 4, 16)

            pred_loss, sigreg_loss, _, _ = model(xb, ab)
            loss = pred_loss + SIGREG_LAMBDA * sigreg_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            e_pred += pred_loss.item() * len(idx)
            e_reg += sigreg_loss.item() * len(idx)
            n += len(idx)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"L_pred={e_pred/n:.4f}  SIGReg={e_reg/n:.4f}")

        if (epoch + 1) % 10 == 0:
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                         "embed_dim": EMBED_DIM}, args.checkpoint)
            print(f"  -> saved: {args.checkpoint}")

    # Save final
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                 "embed_dim": EMBED_DIM}, args.checkpoint)
    print(f"JEPA training done -> {args.checkpoint}")

    # =====================================================
    # Phase 2: State probe (if states available)
    # =====================================================
    if has_states:
        print(f"\n=== State Probe Training: {args.probe_epochs} epochs ===")
        probe = StateProbe(embed_dim=EMBED_DIM, state_dim=states.shape[1]).to(device)
        probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        # Normalize states
        s_mean = win_states.mean(0)
        s_std = win_states.std(0).clamp(min=1e-6)

        model.eval()
        for epoch in range(args.probe_epochs):
            perm = torch.randperm(len(windows))
            e_loss, n = 0.0, 0

            for i in range(0, len(windows), bs):
                idx = perm[i:i + bs]
                xb = windows[idx].to(device)
                sb = ((win_states[idx] - s_mean) / s_std).to(device)

                with torch.no_grad():
                    emb = model.encode(xb)  # (B, 4, D)
                    last_emb = emb[:, -1]   # (B, D) — last frame embedding

                pred_state = probe(last_emb)
                loss = F.mse_loss(pred_state, sb)

                probe_opt.zero_grad()
                loss.backward()
                probe_opt.step()

                e_loss += loss.item() * len(idx)
                n += len(idx)

            print(f"  Probe epoch {epoch+1}/{args.probe_epochs}  L_state={e_loss/n:.4f}")

        # Save with probe
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt["probe"] = probe.state_dict()
        ckpt["state_mean"] = s_mean
        ckpt["state_std"] = s_std
        torch.save(ckpt, args.checkpoint)
        print(f"State probe saved -> {args.checkpoint}")

    # =====================================================
    # Quick latent rollout test
    # =====================================================
    print(f"\n=== Latent Rollout Test ===")
    model.eval()
    with torch.no_grad():
        # Take first 4 frames as seed
        seed = windows[:1].to(device)      # (1, 4, H, W)
        seed_a = win_audio[:1].to(device)  # (1, 4, 16)

        # Rollout 20 steps with zero audio
        future_a = torch.zeros(1, 20, 16, device=device)
        preds = model.rollout(seed, seed_a, future_a, n_steps=20)
        print(f"Rollout: {preds.shape}")  # (1, 20, D)

        # Check if predictions stay in distribution (not collapsed)
        mean = preds.mean().item()
        std = preds.std().item()
        drift = (preds[:, -1] - preds[:, 0]).norm().item()
        print(f"  mean={mean:.4f}  std={std:.4f}  drift={drift:.4f}")

        if has_states:
            # Decode with probe
            probe.eval()
            decoded = probe(preds.reshape(-1, EMBED_DIM)).reshape(1, 20, -1)
            decoded = decoded * s_std.to(device) + s_mean.to(device)
            print(f"  Predicted ball trajectory (y):")
            for t in [0, 4, 9, 14, 19]:
                y = decoded[0, t, 1].item()  # ball_y
                print(f"    t={t:2d}: y={y:.2f}")
