"""Collect Pong training data as HDF5 for stable-worldmodel LeWM training.

Generates episodes with AI-controlled paddles, saves in the HDF5 format
expected by stable_worldmodel.data.dataset.HDF5Dataset.

Usage:
    python scripts/collect_pong_data.py --episodes 1000 --steps 100 --output pong_train
"""

import argparse
import os

import h5py
import numpy as np

from world_model.envs.pong_world import PongWorld


def collect(n_episodes: int, steps_per_ep: int, frameskip: int, image_size: int,
            output_name: str, seed: int = 42):
    env = PongWorld()
    rng = np.random.default_rng(seed)

    all_pixels = []
    all_actions = []
    all_states = []
    ep_lengths = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        noise = rng.uniform(0.0, 0.15)  # varying AI skill

        ep_pixels = []
        ep_actions = []
        ep_states = []

        for step in range(steps_per_ep):
            action = env.ai_action(noise=noise)

            for _ in range(frameskip):
                env.step(action)

            frame = env.render(image_size)  # (H, W, 3) uint8
            state = env.get_state()  # (10,) float32

            ep_pixels.append(frame)
            ep_actions.append(np.array(action, dtype=np.float32))
            ep_states.append(state)

        all_pixels.extend(ep_pixels)
        all_actions.extend(ep_actions)
        all_states.extend(ep_states)
        ep_lengths.append(len(ep_pixels))

        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{n_episodes} "
                  f"(noise={noise:.2f}, score={env.score_l}-{env.score_r})")

    pixels = np.array(all_pixels)    # (N, H, W, 3) uint8
    actions = np.array(all_actions)  # (N, 2) float32
    states = np.array(all_states)    # (N, 10) float32
    ep_lens = np.array(ep_lengths, dtype=np.int32)
    ep_offsets = np.cumsum(np.concatenate([[0], ep_lens[:-1]])).astype(np.int64)
    ep_idx = np.concatenate([np.full(l, i, dtype=np.int32) for i, l in enumerate(ep_lens)])

    # Save as HDF5 in stable-worldmodel format
    datasets_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    datasets_dir = os.path.join(datasets_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    h5_path = os.path.join(datasets_dir, f"{output_name}.h5")

    print(f"\nSaving to {h5_path}")
    print(f"  pixels: {pixels.shape} ({pixels.nbytes / 1e9:.1f} GB)")
    print(f"  actions: {actions.shape}")
    print(f"  states: {states.shape}")
    print(f"  episodes: {len(ep_lens)}")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("pixels", data=pixels, chunks=(1, image_size, image_size, 3))
        f.create_dataset("action", data=actions)
        f.create_dataset("state", data=states)
        f.create_dataset("ep_len", data=ep_lens)
        f.create_dataset("ep_offset", data=ep_offsets)
        f.create_dataset("ep_idx", data=ep_idx)

    print(f"Saved {output_name} ({os.path.getsize(h5_path) / 1e6:.0f} MB)")
    return h5_path


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes", type=int, default=1000)
    pa.add_argument("--steps", type=int, default=100)
    pa.add_argument("--frameskip", type=int, default=5)
    pa.add_argument("--image-size", type=int, default=224)
    pa.add_argument("--output", default="pong_train")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    collect(args.episodes, args.steps, args.frameskip, args.image_size,
            args.output, args.seed)
