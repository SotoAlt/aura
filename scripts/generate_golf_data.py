"""Generate golf-style training data for JEPA world model.

Each episode is a "shot": audio burst for 3-5 frames, then silence.
Ball starts at bottom-left, trajectory depends on shot audio:
  - RMS (volume) = launch power
  - Spectral centroid (pitch) = launch angle
  - Bass = gravity strength

Output: NPZ with frames, states, audios, episodes.
"""
import argparse
import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_model.envs.bounce_world import BounceWorld
from world_model.ascii_model.model import frame_to_indices


def make_shot_audio(power: float, angle: float, rng) -> np.ndarray:
    """Create 16-float audio context for a golf shot.

    power: 0-1 (how hard)
    angle: 0-1 (0=flat, 1=steep)
    """
    ctx = np.zeros(16, dtype=np.float32)
    ctx[0] = ctx[1] = rng.uniform(0.1, 0.5)           # bass (gravity)
    ctx[2] = ctx[3] = rng.uniform(0.2, 0.5)           # mid (wind)
    ctx[4] = ctx[5] = rng.uniform(0.1, 0.4)           # high
    ctx[6] = ctx[7] = min(1.0, power * 1.5)           # onset (trigger)
    ctx[8] = ctx[9] = 0.5                              # bpm (neutral)
    ctx[10] = ctx[11] = angle                          # centroid = angle
    ctx[12] = ctx[13] = power                          # RMS = power
    ctx[14] = ctx[15] = 0.0
    return np.clip(ctx, 0, 1)


def generate_golf_episodes(n_episodes=500, steps_per_ep=100, seed=42, frameskip=5):
    """Generate golf data with frameskip (paper uses frameskip=5).

    Frameskip means we run the physics for `frameskip` steps but only
    record every Nth frame. This makes position changes between recorded
    frames large enough for the probe to detect.

    Paper: frameskip=5, so each "training step" = 5 env steps.
    Ball moves ~3 pixels per training step instead of ~0.6.
    """
    env = BounceWorld()
    rng = np.random.default_rng(seed)

    all_frames, all_states, all_audios, all_episodes = [], [], [], []

    profiles = [
        ("soft_low",   0.3, 0.3),
        ("soft_high",  0.3, 0.8),
        ("mid_low",    0.6, 0.3),
        ("mid_mid",    0.6, 0.5),
        ("mid_high",   0.6, 0.8),
        ("hard_low",   0.9, 0.2),
        ("hard_mid",   0.9, 0.5),
        ("hard_high",  0.9, 0.8),
        ("random",     -1, -1),
        ("gentle",     0.2, 0.6),
    ]

    shot_env_frames = 4 * frameskip  # shot lasts 4 training steps = 20 env frames

    for ep in range(n_episodes):
        profile_name, base_power, base_angle = profiles[ep % len(profiles)]

        if base_power < 0:
            power = rng.uniform(0.15, 0.95)
            angle = rng.uniform(0.1, 0.9)
        else:
            power = base_power + rng.uniform(-0.1, 0.1)
            angle = base_angle + rng.uniform(-0.1, 0.1)

        power = np.clip(power, 0.1, 1.0)
        angle = np.clip(angle, 0.05, 0.95)

        env.reset(seed=seed + ep)
        env.ball_x = env.W * 0.1
        env.ball_y = env.H - 4
        env.vel_x = 0
        env.vel_y = 0

        shot_audio = make_shot_audio(power, angle, rng)
        zero_audio = np.zeros(16, dtype=np.float32)

        total_env_steps = steps_per_ep * frameskip

        for env_step in range(total_env_steps):
            # Shot audio for first shot_env_frames, then silence
            audio = shot_audio.copy() if env_step < shot_env_frames else zero_audio.copy()
            state = env.step(audio)

            # Only record every `frameskip` steps (paper: frameskip=5)
            if env_step % frameskip == 0:
                frame_str = env.render_ascii(audio)
                frame_idx = frame_to_indices(frame_str)
                all_frames.append(frame_idx)
                all_states.append(state)
                # For audio, use the audio that was active during this block
                # (same as paper: actions are "held" for frameskip steps)
                all_audios.append(audio)
                all_episodes.append(ep)

        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{n_episodes} ({profile_name}, "
                  f"power={power:.2f}, angle={angle:.2f}, frameskip={frameskip})")

    return {
        "frames": np.array(all_frames),
        "states": np.array(all_states),
        "audios": np.array(all_audios),
        "episodes": np.array(all_episodes),
    }


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes", type=int, default=500)
    pa.add_argument("--steps", type=int, default=100)
    pa.add_argument("--output", default="data/golf_v1.npz")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--preview", action="store_true")
    args = pa.parse_args()

    if args.preview:
        # Show a few trajectories
        env = BounceWorld()
        rng = np.random.default_rng(42)
        zero = np.zeros(16, dtype=np.float32)

        for name, power, angle in [("soft", 0.3, 0.5), ("medium", 0.6, 0.5), ("hard", 0.9, 0.3)]:
            env.reset(seed=42)
            env.ball_x = env.W * 0.1
            env.ball_y = env.H - 4
            shot = make_shot_audio(power, angle, rng)

            positions = []
            for i in range(40):
                audio = shot if i < 4 else zero
                s = env.step(audio)
                positions.append((s[0] * env.W, s[1] * env.H))

            print(f"\n=== {name} (power={power}, angle={angle}) ===")
            for t in [0, 3, 6, 10, 15, 20, 30, 39]:
                x, y = positions[t]
                print(f"  t={t:2d}: x={x:5.1f} y={y:5.1f}")
    else:
        print(f"Generating {args.episodes} golf episodes x {args.steps} steps...")
        data = generate_golf_episodes(args.episodes, args.steps, args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **data)
        print(f"Saved {len(data['frames'])} frames to {args.output}")
        for k in data:
            print(f"  {k}: {data[k].shape}")
