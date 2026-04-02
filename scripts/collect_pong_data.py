"""Collect Pong training data as HDF5 for stable-worldmodel LeWM training.

Writes directly to HDF5 in chunks — no RAM accumulation.

Usage:
    python scripts/collect_pong_data.py --episodes 3000 --steps 100 --output pong_train_3k
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

    total_frames = n_episodes * steps_per_ep

    datasets_dir = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    datasets_dir = os.path.join(datasets_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    h5_path = os.path.join(datasets_dir, f"{output_name}.h5")

    print(f"Writing {n_episodes} episodes x {steps_per_ep} steps to {h5_path}")
    print(f"  Pre-allocating {total_frames} frames at {image_size}x{image_size}")

    with h5py.File(h5_path, "w") as f:
        # Pre-allocate datasets
        px = f.create_dataset("pixels", shape=(total_frames, image_size, image_size, 3),
                              dtype=np.uint8, chunks=(1, image_size, image_size, 3))
        act = f.create_dataset("action", shape=(total_frames, 2), dtype=np.float32)
        st = f.create_dataset("state", shape=(total_frames, 10), dtype=np.float32)
        ep_len = f.create_dataset("ep_len", shape=(n_episodes,), dtype=np.int32)
        ep_offset = f.create_dataset("ep_offset", shape=(n_episodes,), dtype=np.int64)
        ep_idx_ds = f.create_dataset("ep_idx", shape=(total_frames,), dtype=np.int32)

        idx = 0
        for ep in range(n_episodes):
            env.reset(seed=seed + ep)
            noise = rng.uniform(0.0, 0.15)
            ep_start = idx

            for step in range(steps_per_ep):
                action = env.ai_action(noise=noise)
                for _ in range(frameskip):
                    env.step(action)

                px[idx] = env.render(image_size)
                act[idx] = np.array(action, dtype=np.float32)
                st[idx] = env.get_state()
                ep_idx_ds[idx] = ep
                idx += 1

            ep_len[ep] = steps_per_ep
            ep_offset[ep] = ep_start

            if (ep + 1) % 100 == 0 or ep == 0:
                print(f"  Episode {ep+1}/{n_episodes} "
                      f"(noise={noise:.2f}, score={env.score_l}-{env.score_r})")

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"Saved {output_name}: {total_frames} frames, {size_mb:.0f} MB")
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
