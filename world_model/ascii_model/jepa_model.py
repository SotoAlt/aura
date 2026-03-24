"""JEPA (Joint Embedding Predictive Architecture) for ASCII corridor frames.

Aligned with LeWorldModel (le-wm.github.io):
  - GlyphEncoder: CNN embeds a 40x80 glyph grid into a 192-dim latent
  - LatentPredictor: Transformer with AdaLN-zero audio conditioning per layer
  - GlyphDecoder: Transpose-CNN decodes latent back to glyph logits (trained separately)
  - Loss: L_pred + λ·SIGReg (2-term only, like LeWM)
  - SIGReg: Epps-Pulley characteristic function test for isotropic Gaussian

~3-5M parameters, trains on CPU or GPU.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.ascii_model.model import VOCAB_SIZE

FRAME_H, FRAME_W = 40, 80
CTX_FRAMES = 3
AUDIO_DIM = 16
ACTION_DIM = 2  # [forward, turn]
COND_DIM = AUDIO_DIM + ACTION_DIM  # 18 total conditioning
LATENT_DIM = 192


# ---------------------------------------------------------------------------
# SIGReg — Sketched Isotropic Gaussian Regularizer (from LeWM)
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    """Epps-Pulley characteristic function test for isotropic Gaussian.

    Projects embeddings onto random directions and compares the empirical
    characteristic function against exp(-t²/2). Much stronger than simple
    variance regularization — enforces full Gaussian structure.
    """

    def __init__(self, knots: int = 17, num_proj: int = 512):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) — batch of latent embeddings."""
        if z.dim() == 2:
            z = z.unsqueeze(0)  # (1, B, D) — add time dim
        # Random unit projections
        A = torch.randn(z.size(-1), self.num_proj, device=z.device)
        A = A.div_(A.norm(p=2, dim=0))
        # Epps-Pulley statistic
        x_t = (z @ A).unsqueeze(-1) * self.t  # (..., num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * z.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# AdaLN-Zero Transformer Block (from LeWM / DiT pattern)
# ---------------------------------------------------------------------------

class AdaLNBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning.

    The conditioning signal (audio+action) modulates LayerNorm params in each
    layer, giving much deeper conditioning than FiLM on inputs alone.
    """

    def __init__(self, dim: int, n_heads: int, ff_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )
        # AdaLN-zero: conditioning -> (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )
        # Initialize gate projections to zero for stable training
        nn.init.zeros_(self.adaLN_mlp[-1].weight)
        nn.init.zeros_(self.adaLN_mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask=None):
        """x: (B, S, D), c: (B, cond_dim)"""
        params = self.adaLN_mlp(c).unsqueeze(1)  # (B, 1, 6*D)
        s1, sh1, g1, s2, sh2, g2 = params.chunk(6, dim=-1)

        # Attention with AdaLN
        h = self.norm1(x) * (1 + s1) + sh1
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=(mask is not None))
        x = x + g1 * h

        # FFN with AdaLN
        h = self.norm2(x) * (1 + s2) + sh2
        h = self.ff(h)
        x = x + g2 * h

        return x


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
# 2. LatentPredictor — AdaLN-zero Transformer with audio in every layer
# ---------------------------------------------------------------------------

class LatentPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        cond_dim: int = COND_DIM,
        n_ctx: int = CTX_FRAMES,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_ctx = n_ctx

        # Conditioning projection: audio(16) + action(2) -> latent_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.SiLU(), nn.Linear(128, latent_dim),
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Embedding(n_ctx, latent_dim)

        # AdaLN-zero transformer blocks — audio+action conditions EVERY layer
        self.blocks = nn.ModuleList([
            AdaLNBlock(latent_dim, n_heads, ff_dim, latent_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(latent_dim)

        # Projection head for predicted latent
        self.head = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents: torch.Tensor, audio: torch.Tensor,
                action: torch.Tensor = None) -> torch.Tensor:
        """
        latents: (B, n_ctx, latent_dim)
        audio:   (B, audio_dim)
        action:  (B, action_dim) or None — [forward, turn]
        returns: (B, latent_dim) predicted next latent
        """
        B, S, D = latents.shape

        # Build conditioning: concat audio + action
        if action is not None:
            cond_input = torch.cat([audio, action], dim=1)  # (B, 18)
        else:
            # Backward compat: no action = stand still
            cond_input = torch.cat([
                audio,
                torch.zeros(B, ACTION_DIM, device=audio.device)
            ], dim=1)  # (B, 18)

        c = self.cond_proj(cond_input)  # (B, latent_dim)

        # Add positional embeddings
        positions = torch.arange(S, device=latents.device)
        x = latents + self.pos_embed(positions).unsqueeze(0)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)

        # Pass through AdaLN-zero blocks — audio modulates every layer
        for block in self.blocks:
            x = block(x, c, mask=mask)

        x = self.final_norm(x)

        # Take last position as prediction
        pred = self.head(x[:, -1])  # (B, D)
        return pred


# ---------------------------------------------------------------------------
# 3. GlyphDecoder — latent (B,192) -> logits (B, VOCAB_SIZE, 40, 80)
# ---------------------------------------------------------------------------

class GlyphDecoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, audio_dim: int = AUDIO_DIM):
        super().__init__()
        self.project = nn.Linear(latent_dim + audio_dim, 128 * 5 * 10)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, VOCAB_SIZE, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor, audio: torch.Tensor = None) -> torch.Tensor:
        """z: (B, latent_dim), audio: (B, 16) -> (B, VOCAB_SIZE, 40, 80)"""
        if audio is not None:
            z = torch.cat([z, audio], dim=1)
        else:
            z = torch.cat([z, torch.zeros(z.shape[0], AUDIO_DIM, device=z.device)], dim=1)
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

    def forward(self, frames_seq: torch.Tensor, audio: torch.Tensor,
                action: torch.Tensor = None):
        B = frames_seq.shape[0]
        ctx_frames = frames_seq[:, :CTX_FRAMES]      # (B, 3, 40, 80)
        target_frame = frames_seq[:, CTX_FRAMES]      # (B, 40, 80)

        # Encode each context frame
        ctx_latents = []
        for i in range(CTX_FRAMES):
            ctx_latents.append(self.encoder(ctx_frames[:, i]))
        ctx_latents = torch.stack(ctx_latents, dim=1)  # (B, 3, D)

        # Encode target frame — NO stop-gradient (LeWM/LeJEPA: SIGReg prevents
        # collapse without needing stop-grad or teacher-student)
        target_latent = self.encoder(target_frame)

        # Predict next latent from context + audio + action
        predicted_latent = self.predictor(ctx_latents, audio, action)

        # Decode predicted latent + audio to glyph logits
        decoded_logits = self.decoder(predicted_latent, audio)

        return predicted_latent, target_latent, decoded_logits

    # Keep backward compat alias
    @staticmethod
    def variance_regularization(latents: torch.Tensor, min_std: float = 0.1) -> torch.Tensor:
        """Legacy — use SIGReg instead."""
        std = latents.std(dim=0)
        penalty = F.relu(min_std - std).pow(2).mean()
        return penalty


# ---------------------------------------------------------------------------
# 5. Training script — 2-phase like LeWM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train AsciiJEPA (LeWM-aligned)")
    pa.add_argument("--data", default="data/ascii_training.npz")
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--checkpoint", default="checkpoints/ascii_jepa.pt")
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--sigreg-lambda", type=float, default=10.0,
                    help="SIGReg weight (LeWM default ~10)")
    pa.add_argument("--decoder-phase-frac", type=float, default=0.3,
                    help="Last 30%% of epochs: freeze JEPA, train decoder only")
    args = pa.parse_args()

    # --- Load data ---
    d = np.load(args.data)
    frames = torch.from_numpy(d["frames"]).long()     # (N, 40, 80)
    audio = torch.from_numpy(d["audios"]).float()      # (N, 16)
    has_actions = "actions" in d
    if has_actions:
        actions = torch.from_numpy(d["actions"]).float()  # (N, 2)
        episodes = d.get("episodes", None)
    else:
        actions = torch.zeros(len(frames), ACTION_DIM)
        episodes = None

    # Build 4-frame windows respecting episode boundaries
    n_ctx = CTX_FRAMES
    valid_indices = []
    for i in range(n_ctx, len(frames)):
        # If we have episode info, skip cross-episode windows
        if episodes is not None and episodes[i] != episodes[i - n_ctx]:
            continue
        valid_indices.append(i)
    valid_indices = torch.tensor(valid_indices)

    seq = torch.stack([frames[i - n_ctx : i + 1] for i in valid_indices])
    aud = audio[valid_indices]
    act = actions[valid_indices]  # action that produced frame[i]
    tgt = frames[valid_indices]
    print(f"Dataset: {len(seq)} samples from {len(frames)} frames"
          f" (actions={'yes' if has_actions else 'no'})")

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
    sigreg = SIGReg().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AsciiJEPA: {n_params:,} parameters on {device}")

    # Phase boundary
    jepa_epochs = int(args.epochs * (1 - args.decoder_phase_frac))
    decoder_start = jepa_epochs
    print(f"Phase 1 (JEPA): epochs 1-{jepa_epochs}  |  "
          f"Phase 2 (decoder): epochs {jepa_epochs + 1}-{args.epochs}")

    # Separate optimizers for 2-phase training
    jepa_params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    decoder_params = list(model.decoder.parameters())
    opt_jepa = torch.optim.AdamW(jepa_params, lr=1e-3)
    opt_decoder = torch.optim.AdamW(decoder_params, lr=1e-3)

    bs = args.batch_size
    step = 0

    for epoch in range(args.epochs):
        perm = torch.randperm(len(seq))
        e_pred, e_dec, e_reg, e_correct, e_total = 0.0, 0.0, 0.0, 0, 0
        phase2 = epoch >= decoder_start

        if phase2 and epoch == decoder_start:
            print("\n=== Phase 2: Freeze encoder+predictor, train decoder ===")
            model.encoder.eval()
            model.predictor.eval()
            for p in jepa_params:
                p.requires_grad_(False)

        for i in range(0, len(seq), bs):
            idx = perm[i : i + bs]
            xb = seq[idx].to(device)
            ab = aud[idx].to(device)
            actb = act[idx].to(device)
            yb = tgt[idx].to(device)

            # AR rollout: in JEPA phase after 60%, in decoder phase always
            ar_active = (not phase2 and epoch >= int(jepa_epochs * 0.6)) or phase2
            use_ar = ar_active and (step % 2 == 0)

            if use_ar:
                model.eval()
                with torch.no_grad():
                    p_lat, _, p_logits = model(xb, ab, actb)
                    pred_frame = p_logits.argmax(dim=1)
                    ar_xb = torch.cat([xb[:, 1:], pred_frame.unsqueeze(1)], dim=1)
                model.train()
                if phase2:
                    model.encoder.eval()
                    model.predictor.eval()
                pred_lat, tgt_lat, logits = model(ar_xb, ab, actb)
            else:
                pred_lat, tgt_lat, logits = model(xb, ab, actb)

            if not phase2:
                # Phase 1: L_pred + λ·SIGReg ONLY (no decoder loss!)
                l_pred = F.mse_loss(pred_lat, tgt_lat)
                l_sigreg = sigreg(pred_lat)
                loss = l_pred + args.sigreg_lambda * l_sigreg

                # Track decoder loss for logging only (no gradients)
                with torch.no_grad():
                    l_decode = F.cross_entropy(logits, yb)

                opt_jepa.zero_grad()
                loss.backward()
                opt_jepa.step()
            else:
                # Phase 2: L_decode ONLY (encoder+predictor frozen)
                with torch.no_grad():
                    l_pred = F.mse_loss(pred_lat, tgt_lat)
                    l_sigreg = sigreg(pred_lat)
                l_decode = F.cross_entropy(logits, yb)
                loss = l_decode

                opt_decoder.zero_grad()
                loss.backward()
                opt_decoder.step()

            # Accuracy
            with torch.no_grad():
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
                phase_str = "DEC" if phase2 else "JEPA"
                print(
                    f"  [{phase_str}] step {step:>5d}  "
                    f"L_pred={l_pred.item():.4f}  "
                    f"L_dec={l_decode.item():.4f}  "
                    f"SIGReg={l_sigreg.item():.4f}  "
                    f"acc={correct / total * 100:.1f}%"
                )

        n = len(seq)
        phase_str = "DEC" if phase2 else "JEPA"
        print(
            f"[{phase_str}] Epoch {epoch + 1}/{args.epochs}  "
            f"L_pred={e_pred / n:.4f}  "
            f"L_dec={e_dec / n:.4f}  "
            f"SIGReg={e_reg / n:.4f}  "
            f"acc={e_correct / e_total * 100:.1f}%"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "vocab_size": VOCAB_SIZE,
                 "epoch": epoch + 1, "phase": "decoder" if phase2 else "jepa"},
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

    # --- Quick sample: generate frames with different audio contexts ---
    print("\n=== Autoregressive Sample (varied audio) ===")
    model.eval()
    with torch.no_grad():
        seed = frames[:4].to(device)
        scenarios = {
            'high_energy': torch.tensor([[0.9]*2 + [0.9]*2 + [0.5]*2 + [0.0]*2 + [0.6]*2 + [0.5]*2 + [0.9]*2 + [0.0]*2], dtype=torch.float32),
            'low_energy': torch.tensor([[0.05]*2 + [0.05]*2 + [0.05]*2 + [0.0]*2 + [0.3]*2 + [0.3]*2 + [0.05]*2 + [0.0]*2], dtype=torch.float32),
            'onset_burst': torch.tensor([[0.5]*2 + [0.5]*2 + [0.5]*2 + [1.0]*2 + [0.6]*2 + [0.5]*2 + [0.8]*2 + [0.0]*2], dtype=torch.float32),
        }
        for scenario_name, aud_vec in scenarios.items():
            print(f"\n=== {scenario_name} ===")
            buf = [seed[i] for i in range(4)]
            for s in range(5):
                ctx = torch.stack(buf[-3:]).unsqueeze(0).to(device)
                tgt_dummy = buf[-1].unsqueeze(0).to(device)
                inp = torch.cat([ctx, tgt_dummy.unsqueeze(1)], dim=1)
                a = aud_vec.to(device)

                ctx_lats = torch.stack([model.encoder(inp[0, j].unsqueeze(0)) for j in range(3)], dim=1)
                pred_lat = model.predictor(ctx_lats, a)
                logits = model.decoder(pred_lat, a)
                pred_frame = logits.argmax(dim=1)[0]
                buf.append(pred_frame)

                from world_model.ascii_model.model import indices_to_frame
                print(f"\n--- {scenario_name} Frame {s + 1} ---")
                print(indices_to_frame(pred_frame.cpu().numpy()))
