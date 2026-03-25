"""JEPA Hybrid World Model: JEPA dynamics + raycaster rendering.

The JEPA learns dynamics from visual observations (encoding ASCII frames into
latent space, predicting next latent from audio+action). Instead of decoding
back to noisy frames, we decode to SCENE STATE parameters and render with the
raycaster for clean, sharp output.

Architecture:
    GlyphEncoder: ASCII frame (40x80) → 192/256-dim latent
    LatentPredictor: (past latents + audio + action) → next latent
    StateDecoder: latent → [pos_x, pos_y, angle, audio_params(16)] = 19 floats
    Raycaster: state → clean ASCII frame @ 60 FPS

Training:
    Phase 1: Encoder + Predictor with L_pred + SIGReg (learn dynamics)
    Phase 2: StateDecoder with MSE on state params (learn to extract state)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.ascii_model.jepa_model import (
    AsciiJEPA, SIGReg, GlyphEncoder, LatentPredictor,
    LATENT_DIM, AUDIO_DIM, ACTION_DIM, COND_DIM,
)

STATE_DIM = 19  # pos_x, pos_y, angle, audio_context(16)


class StateDecoder(nn.Module):
    """Decode latent → scene state parameters for the raycaster."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, STATE_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, 19) scene state."""
        return self.net(z)


class JEPAHybrid(nn.Module):
    """JEPA with raycaster rendering.

    Uses the standard JEPA encoder+predictor for dynamics,
    but outputs scene state instead of frame glyphs.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_ctx: int = 3,
                 n_layers: int = 4, ff_dim: int = 384):
        super().__init__()
        self.encoder = GlyphEncoder(latent_dim=latent_dim)
        self.predictor = LatentPredictor(
            latent_dim=latent_dim, n_ctx=n_ctx,
            n_layers=n_layers, ff_dim=ff_dim,
        )
        self.state_decoder = StateDecoder(latent_dim=latent_dim)

    def forward(self, frames_seq: torch.Tensor, audio: torch.Tensor,
                action: torch.Tensor = None):
        """
        frames_seq: (B, n_ctx+1, 40, 80) — context + target
        audio: (B, 16)
        action: (B, 2) or None
        Returns: predicted_latent, target_latent, predicted_state
        """
        B = frames_seq.shape[0]
        n_ctx = frames_seq.shape[1] - 1
        ctx_frames = frames_seq[:, :n_ctx]
        target_frame = frames_seq[:, n_ctx]

        ctx_latents = torch.stack(
            [self.encoder(ctx_frames[:, i]) for i in range(n_ctx)], dim=1
        )
        target_latent = self.encoder(target_frame)
        predicted_latent = self.predictor(ctx_latents, audio, action)
        predicted_state = self.state_decoder(predicted_latent)

        return predicted_latent, target_latent, predicted_state

    def predict_state(self, ctx_latents: torch.Tensor, audio: torch.Tensor,
                      action: torch.Tensor = None) -> torch.Tensor:
        """Direct latent → state for inference (no frame encoding)."""
        pred_latent = self.predictor(ctx_latents, audio, action)
        return self.state_decoder(pred_latent), pred_latent


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train JEPA Hybrid (JEPA + raycaster)")
    pa.add_argument("--data", required=True)
    pa.add_argument("--epochs", type=int, default=60)
    pa.add_argument("--batch-size", type=int, default=64)
    pa.add_argument("--checkpoint", default="checkpoints/jepa_hybrid.pt")
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--sigreg-lambda", type=float, default=1.0)
    pa.add_argument("--large", action="store_true")
    pa.add_argument("--ctx-frames", type=int, default=None)
    args = pa.parse_args()

    # Config
    if args.large:
        lat_dim, n_layers, ff_dim = 256, 6, 512
        n_ctx = args.ctx_frames or 5
    else:
        lat_dim, n_layers, ff_dim = LATENT_DIM, 4, 384
        n_ctx = args.ctx_frames or 3

    # Load data
    d = np.load(args.data)
    frames = torch.from_numpy(d["frames"]).long()
    audio = torch.from_numpy(d["audios"]).float()
    actions = torch.from_numpy(d["actions"]).float() if "actions" in d else torch.zeros(len(frames), 2)
    states = torch.from_numpy(d["states"]).float() if "states" in d else None
    episodes = d.get("episodes", None)

    if states is None:
        print("ERROR: Training data needs 'states' (pos_x, pos_y, angle).")
        print("Regenerate with updated generate_wm_data.py")
        raise SystemExit(1)

    # Build state targets: [pos_x, pos_y, angle, audio(16)] = 19 floats
    state_targets = torch.cat([states, audio], dim=1)  # (N, 19)

    # Normalize state targets to [0, 1] range for stable training
    state_min = state_targets.min(dim=0).values
    state_max = state_targets.max(dim=0).values
    state_range = (state_max - state_min).clamp(min=1e-6)
    state_targets_norm = (state_targets - state_min) / state_range

    # Build windows
    valid = []
    for i in range(n_ctx, len(frames)):
        if episodes is not None and episodes[i] != episodes[i - n_ctx]:
            continue
        valid.append(i)
    valid = torch.tensor(valid)

    seq = torch.stack([frames[i - n_ctx : i + 1] for i in valid])
    aud = audio[valid]
    act = actions[valid]
    tgt_state = state_targets_norm[valid]
    print(f"Dataset: {len(seq)} samples, n_ctx={n_ctx}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Model
    model = JEPAHybrid(latent_dim=lat_dim, n_ctx=n_ctx,
                        n_layers=n_layers, ff_dim=ff_dim).to(device)
    sigreg = SIGReg().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"JEPAHybrid: {n_params:,} params on {device}")

    # Phase split: 60% dynamics, 40% state decoder
    phase1_epochs = int(args.epochs * 0.6)
    print(f"Phase 1 (dynamics): epochs 1-{phase1_epochs}  |  "
          f"Phase 2 (state decoder): {phase1_epochs+1}-{args.epochs}")

    # Optimizers
    dynamics_params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    state_dec_params = list(model.state_decoder.parameters())
    opt_dynamics = torch.optim.AdamW(dynamics_params, lr=1e-3)
    opt_state = torch.optim.AdamW(state_dec_params, lr=1e-3)

    bs = args.batch_size
    step = 0

    for epoch in range(args.epochs):
        perm = torch.randperm(len(seq))
        e_pred, e_state, e_reg = 0.0, 0.0, 0.0
        phase2 = epoch >= phase1_epochs

        if phase2 and epoch == phase1_epochs:
            print("\n=== Phase 2: Freeze encoder+predictor, train state decoder ===")
            model.encoder.eval()
            model.predictor.eval()
            for p in dynamics_params:
                p.requires_grad_(False)

        for i in range(0, len(seq), bs):
            idx = perm[i : i + bs]
            xb = seq[idx].to(device)
            ab = aud[idx].to(device)
            actb = act[idx].to(device)
            sb = tgt_state[idx].to(device)

            pred_lat, tgt_lat, pred_state = model(xb, ab, actb)

            if not phase2:
                # Phase 1: dynamics only
                l_pred = F.mse_loss(pred_lat, tgt_lat)
                l_reg = sigreg(pred_lat)
                loss = l_pred + args.sigreg_lambda * l_reg
                with torch.no_grad():
                    l_state = F.mse_loss(pred_state, sb)

                opt_dynamics.zero_grad()
                loss.backward()
                opt_dynamics.step()
            else:
                # Phase 2: state decoder only
                with torch.no_grad():
                    l_pred = F.mse_loss(pred_lat, tgt_lat)
                    l_reg = sigreg(pred_lat)
                l_state = F.mse_loss(pred_state, sb)
                loss = l_state

                opt_state.zero_grad()
                loss.backward()
                opt_state.step()

            e_pred += l_pred.item() * len(idx)
            e_state += l_state.item() * len(idx)
            e_reg += l_reg.item() * len(idx)
            step += 1

            if step % 50 == 0:
                tag = "STATE" if phase2 else "DYN"
                print(f"  [{tag}] step {step:>5d}  "
                      f"L_pred={l_pred.item():.4f}  "
                      f"L_state={l_state.item():.4f}  "
                      f"SIGReg={l_reg.item():.4f}")

        n = len(seq)
        tag = "STATE" if phase2 else "DYN"
        print(f"[{tag}] Epoch {epoch+1}/{args.epochs}  "
              f"L_pred={e_pred/n:.4f}  L_state={e_state/n:.4f}  SIGReg={e_reg/n:.4f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "config": {"latent_dim": lat_dim, "n_ctx": n_ctx,
                           "n_layers": n_layers, "ff_dim": ff_dim},
                "state_min": state_min, "state_max": state_max,
                "state_range": state_range,
                "epoch": epoch + 1,
            }, args.checkpoint)
            print(f"  -> saved: {args.checkpoint}")

    # Final save
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {"latent_dim": lat_dim, "n_ctx": n_ctx,
                   "n_layers": n_layers, "ff_dim": ff_dim},
        "state_min": state_min, "state_max": state_max,
        "state_range": state_range,
        "epoch": args.epochs,
    }, args.checkpoint)
    print(f"Saved final checkpoint -> {args.checkpoint}")
