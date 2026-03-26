"""State-space world model: direct state → state prediction with audio.

Instead of JEPA's lossy image→latent→probe→state chain, this operates
directly on physics state vectors. The neural network learns:

    (state_{t-2}, state_{t-1}, state_t, audio_t) → delta_state_{t+1}

Where state = [ball_x, ball_y, vel_x, vel_y, gravity] (5 floats)
and audio = 16-float context vector.

Predicts DELTAS (change in state), not absolute positions.
This prevents mean-regression and enables clean long rollouts.

~100K params. Trains in 3 minutes on CPU.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DIM = 5   # ball_x, ball_y, vel_x, vel_y, gravity
AUDIO_DIM = 16
HISTORY = 3     # number of past states used


class StateDynamicsModel(nn.Module):
    """Physics-informed state dynamics model.

    Base physics: x += vx*dt, y += vy*dt + 0.5*g*dt^2, vy += g*dt
    Neural net learns: audio→force mapping + collision corrections

    The model doesn't need to learn gravity — that's built in.
    It only learns what the physics engine does differently from
    simple ballistic motion (bouncing, audio forces, friction).
    """

    def __init__(self, state_dim=STATE_DIM, audio_dim=AUDIO_DIM,
                 history=HISTORY, hidden=128):
        super().__init__()
        self.history = history
        self.dt = 1.0  # timestep

        # Audio → force network (small, focused)
        self.audio_force = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [force_x, force_y]
        )

        # Correction network: learns bounce, friction, wall collision
        input_dim = state_dim * history + audio_dim
        self.correction = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, state_dim),  # correction to base physics
        )
        # Initialize correction to near-zero (trust physics first)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, state_history, audio):
        """
        state_history: (B, history, 5) — [x, y, vx, vy, gravity]
        audio: (B, 16)
        Returns: (B, 5) — predicted DELTA state
        """
        if state_history.dim() == 2:
            # Flat input: reshape
            state_history = state_history.reshape(-1, self.history, STATE_DIM)

        current = state_history[:, -1]  # (B, 5) — most recent state
        x, y, vx, vy, g = current.unbind(1)

        # Base physics (ballistic motion)
        audio_force = self.audio_force(audio)  # (B, 2)
        fx, fy = audio_force.unbind(1)

        # Physics equations
        dx = vx * self.dt
        dy = vy * self.dt + 0.5 * g * self.dt ** 2
        dvx = fx  # audio force on x
        dvy = g * self.dt + fy  # gravity + audio force on y
        dg = torch.zeros_like(g)  # gravity doesn't change

        base_delta = torch.stack([dx, dy, dvx, dvy, dg], dim=1)  # (B, 5)

        # Neural correction (learns bounce, friction, walls)
        flat_hist = state_history.reshape(state_history.shape[0], -1)
        correction = self.correction(torch.cat([flat_hist, audio], dim=1))

        return base_delta + correction


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train state dynamics world model")
    pa.add_argument("--data", required=True)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--checkpoint", default="checkpoints/state_model.pt")
    pa.add_argument("--hidden", type=int, default=256)
    args = pa.parse_args()

    # Load data
    d = np.load(args.data)
    states = torch.from_numpy(d["states"]).float()   # (N, 5)
    audio = torch.from_numpy(d["audios"]).float()     # (N, 16)
    episodes = d["episodes"]

    print(f"Data: {len(states)} frames, states={states.shape}, audio={audio.shape}")

    # Normalize states for stable training
    s_mean = states.mean(0)
    s_std = states.std(0).clamp(min=1e-6)
    states_norm = (states - s_mean) / s_std

    # Build training windows: 3 past states + current audio → delta to next state
    hist = HISTORY
    X_states, X_audio, Y_delta = [], [], []

    for i in range(hist, len(states) - 1):
        # Skip episode boundaries
        if episodes[i] != episodes[i - hist] or episodes[i] != episodes[i + 1]:
            continue

        past = states_norm[i - hist: i]  # (3, 5) normalized
        aud = audio[i]                    # (16,) raw
        # Delta: normalized difference
        delta = states_norm[i + 1] - states_norm[i]

        X_states.append(past)
        X_audio.append(aud)
        Y_delta.append(delta)

    X_states = torch.stack(X_states)  # (N, 3, 5)
    X_audio = torch.stack(X_audio)    # (N, 16)
    Y_delta = torch.stack(Y_delta)    # (N, 5)

    print(f"Training samples: {len(X_states)}")
    print(f"Delta stats: mean={Y_delta.mean(0).tolist()}, "
          f"std={Y_delta.std(0).tolist()}")

    # Train/val split
    n_val = len(X_states) // 10
    perm = torch.randperm(len(X_states))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Model
    model = StateDynamicsModel(hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"StateDynamicsModel: {n_params:,} params")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    bs = args.batch_size

    best_val = float('inf')

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_perm = train_idx[torch.randperm(len(train_idx))]
        train_loss = 0.0

        for i in range(0, len(train_perm), bs):
            idx = train_perm[i:i + bs]
            pred = model(X_states[idx], X_audio[idx])
            loss = F.mse_loss(pred, Y_delta[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(idx)

        train_loss /= len(train_idx)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_states[val_idx], X_audio[val_idx])
            val_loss = F.mse_loss(val_pred, Y_delta[val_idx]).item()

        scheduler.step(val_loss)
        lr = opt.param_groups[0]['lr']

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  lr={lr:.1e}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "state_mean": s_mean,
                "state_std": s_std,
                "hidden": args.hidden,
                "history": HISTORY,
            }, args.checkpoint)

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved: {args.checkpoint}")

    # =====================================================
    # Validation: 50-step rollout comparison
    # =====================================================
    print("\n=== Rollout Validation ===")

    # Load best model
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Pick a launch episode: find one with onset in first few frames
    from world_model.envs.bounce_world import BounceWorld
    env = BounceWorld()
    env.reset(seed=99)
    env.ball_x = 40
    env.ball_y = 36

    # Run real physics: launch then silence
    launch_audio = np.zeros(16, dtype=np.float32)
    launch_audio[12] = launch_audio[13] = 0.7
    launch_audio[6] = launch_audio[7] = 0.8
    zero_audio = np.zeros(16, dtype=np.float32)

    real_states = []
    for i in range(53):
        audio = launch_audio if i == 0 else zero_audio
        s = env.step(audio)
        real_states.append(s.copy())
    real_states = np.array(real_states)

    # Model rollout from same initial conditions
    with torch.no_grad():
        # Normalize initial states
        init_states_raw = torch.tensor(real_states[:hist], dtype=torch.float32)
        init_states = (init_states_raw - s_mean) / s_std  # (3, 5) normalized

        state_buf = list(init_states.unbind(0))  # 3 normalized states
        predicted_raw = []

        for i in range(50):
            audio_t = torch.tensor(launch_audio if i == 0 else zero_audio).unsqueeze(0)
            hist_t = torch.stack(state_buf[-HISTORY:]).unsqueeze(0)  # (1, 3, 5)

            delta = model(hist_t, audio_t)  # (1, 5)
            next_state = state_buf[-1] + delta[0]
            state_buf.append(next_state)

            # Denormalize for comparison
            raw = next_state * s_std + s_mean
            predicted_raw.append(raw.numpy())

    predicted_raw = np.array(predicted_raw)

    # Compare
    print(f"{'frame':>5s} | {'real_y':>8s} | {'pred_y':>8s} | {'error':>8s}")
    print("-" * 40)
    max_err = 0
    for t in [0, 2, 5, 10, 15, 20, 30, 40, 49]:
        ry = real_states[t + hist, 1]
        py = predicted_raw[t, 1]
        err = abs(ry - py)
        max_err = max(max_err, err)
        # Convert to screen rows (40 rows)
        print(f"  {t:3d}  | {ry*40:8.2f} | {py*40:8.2f} | {err*40:7.2f} rows")

    print(f"\nMax error: {max_err*40:.2f} rows (threshold: 2.0)")
    if max_err * 40 < 2.0:
        print("✅ PASSED — physics prediction is accurate!")
    else:
        print(f"⚠️  Error exceeds threshold. May need more data or epochs.")
