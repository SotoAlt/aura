"""Generate world model training data: (frame, action, audio) tuples.

Records episodes of an agent moving through an ASCII raycaster corridor.
Actions are continuous [forward, turn] so the model learns spatial dynamics.
Audio modulates visual style independently of movement.

Output: NPZ file with:
  frames: (N, 40, 80) int — glyph indices per frame
  actions: (N, 2) float32 — [forward, turn] action that produced this frame
  audios: (N, 16) float32 — audio context for this frame
  episodes: (N,) int — episode index for boundary tracking
"""
import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_model.envs.ascii_corridor import (
    AsciiCorridorEnv,
    _make_audio_context,
    FORWARD, TURN_LEFT, TURN_RIGHT,
)
from world_model.ascii_model.model import frame_to_indices


def generate_episodes(
    n_episodes: int = 100,
    steps_per_ep: int = 200,
    audio_profiles: list[str] | None = None,
    seed: int = 42,
    movement_style: str = "mixed",
) -> dict:
    """Generate world model data with actions.

    Movement styles:
        'forward': mostly forward with occasional turns
        'explore': balanced forward/turn/backward
        'mixed': random mix of styles per episode
    """
    if audio_profiles is None:
        audio_profiles = ["high", "low", "ramp", "pulse", "random", "sweep"]

    env = AsciiCorridorEnv(cols=80, rows=40, map_size=32, max_steps=steps_per_ep)
    rng = np.random.default_rng(seed)

    all_frames = []
    all_actions = []
    all_audios = []
    all_episodes = []
    all_states = []  # [pos_x, pos_y, angle] — scene state for renderer

    for ep in range(n_episodes):
        # Pick audio profile for this episode
        profile = audio_profiles[ep % len(audio_profiles)]

        # Pick movement style
        if movement_style == "mixed":
            style = rng.choice(["forward", "explore", "wander"])
        else:
            style = movement_style

        obs, _ = env.reset(seed=seed + ep)
        ep_map = env._map.copy()  # save map for this episode
        ctx = np.zeros(16, dtype=np.float32)

        for step_i in range(steps_per_ep):
            # Generate audio context
            ctx = _make_audio_context(profile, step_i, steps_per_ep, rng, ctx)
            env.set_context(ctx)

            # Choose action based on movement style
            if style == "forward":
                # Mostly forward, occasional turns
                r = rng.random()
                if r < 0.65:
                    action = FORWARD
                elif r < 0.80:
                    action = TURN_LEFT
                else:
                    action = TURN_RIGHT
            elif style == "explore":
                # Balanced exploration
                action = rng.choice([FORWARD, TURN_LEFT, TURN_RIGHT],
                                     p=[0.4, 0.3, 0.3])
            else:  # wander
                # Random walk with momentum
                if step_i > 0 and rng.random() < 0.6:
                    action = prev_action  # repeat last action
                else:
                    action = rng.choice([FORWARD, TURN_LEFT, TURN_RIGHT])

            prev_action = action

            # Convert discrete action to continuous [forward, turn]
            if action == FORWARD:
                action_vec = np.array([1.0, 0.0], dtype=np.float32)
            elif action == TURN_LEFT:
                action_vec = np.array([0.0, -1.0], dtype=np.float32)
            elif action == TURN_RIGHT:
                action_vec = np.array([0.0, 1.0], dtype=np.float32)

            obs, _, done, truncated, _ = env.step(action)

            # Convert ASCII frame to glyph indices
            frame_indices = frame_to_indices(obs["ascii"])

            # Record scene state for renderer-based decoding
            state_vec = np.array([
                env._pos[0], env._pos[1], env._angle
            ], dtype=np.float32)

            all_frames.append(frame_indices)
            all_actions.append(action_vec)
            all_audios.append(ctx.copy())
            all_episodes.append(ep)
            all_states.append(state_vec)

            if done or truncated:
                break

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{n_episodes} ({profile}, {style})",
                  file=sys.stderr)

    return {
        "frames": np.array(all_frames),    # (N, 40, 80)
        "actions": np.array(all_actions),   # (N, 2)
        "audios": np.array(all_audios),     # (N, 16)
        "episodes": np.array(all_episodes), # (N,)
        "states": np.array(all_states),     # (N, 3) — [pos_x, pos_y, angle]
    }


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Generate world model training data")
    pa.add_argument("--episodes", type=int, default=200)
    pa.add_argument("--steps", type=int, default=200)
    pa.add_argument("--output", type=str, default="data/ascii_wm_v1.npz")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    print(f"Generating {args.episodes} episodes x {args.steps} steps...",
          file=sys.stderr)
    data = generate_episodes(
        n_episodes=args.episodes,
        steps_per_ep=args.steps,
        seed=args.seed,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Saved {len(data['frames'])} frames to {args.output}", file=sys.stderr)
    print(f"  frames:   {data['frames'].shape}", file=sys.stderr)
    print(f"  actions:  {data['actions'].shape}", file=sys.stderr)
    print(f"  audios:   {data['audios'].shape}", file=sys.stderr)
    print(f"  episodes: {data['episodes'].shape}", file=sys.stderr)
