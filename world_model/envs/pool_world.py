"""2D Pool table physics engine for JEPA world model training.

7 balls (1 cue + 6 colored) with elastic collisions, wall bounces,
friction, and pockets. Renders as 128x128 RGB images.

The JEPA learns to predict multi-body chain reactions — dynamics that
a simple formula can't capture (ball→ball→ball collisions).

Usage:
    env = PoolWorld()
    env.reset()
    env.shoot(angle=0.5, power=0.8)  # hit cue ball
    for _ in range(60):
        env.step()
        img = env.render()  # (128, 128, 3) uint8
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field


# Table dimensions (normalized 0-1)
TABLE_W = 1.0
TABLE_H = 0.5
BALL_RADIUS = 0.018
POCKET_RADIUS = 0.035
FRICTION = 0.985  # velocity multiplier per step
MIN_VEL = 0.0005  # below this, ball stops

# Pocket positions (6 pockets)
POCKETS = [
    (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),  # top
    (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),  # bottom
]

# Ball colors (RGB) for rendering
BALL_COLORS = [
    (255, 255, 255),  # 0: cue (white)
    (220, 40, 40),    # 1: red
    (40, 40, 220),    # 2: blue
    (220, 180, 30),   # 3: yellow
    (40, 180, 40),    # 4: green
    (180, 50, 180),   # 5: purple
    (220, 120, 30),   # 6: orange
]

TABLE_COLOR = (30, 100, 50)     # green felt
BORDER_COLOR = (60, 40, 20)     # brown border
POCKET_COLOR = (15, 15, 15)     # dark pockets


@dataclass
class Ball:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    color_idx: int = 0
    active: bool = True  # False if pocketed


class PoolWorld:
    """2D pool table physics simulation."""

    def __init__(self, n_balls: int = 7):
        self.n_balls = n_balls
        self.balls: list[Ball] = []
        self.rng = np.random.default_rng()
        self.pocketed: list[int] = []

    def reset(self, seed=None):
        """Reset table with balls in triangle formation."""
        self.rng = np.random.default_rng(seed)
        self.balls = []
        self.pocketed = []

        # Cue ball on left side
        self.balls.append(Ball(x=0.25, y=0.25, color_idx=0))

        # Other balls in loose triangle on right side
        start_x = 0.65
        start_y = 0.25
        spacing = BALL_RADIUS * 2.5
        idx = 1
        for row in range(3):
            for col in range(row + 1):
                if idx >= self.n_balls:
                    break
                bx = start_x + row * spacing * 0.866  # cos(30°)
                by = start_y + (col - row / 2) * spacing
                # Add small random offset for variety
                bx += self.rng.uniform(-0.005, 0.005)
                by += self.rng.uniform(-0.005, 0.005)
                self.balls.append(Ball(x=bx, y=by, color_idx=idx))
                idx += 1

        return self.get_state()

    def shoot(self, angle: float, power: float):
        """Hit the cue ball. angle in radians, power 0-1."""
        if not self.balls[0].active:
            return
        speed = power * 0.08  # max speed
        self.balls[0].vx = math.cos(angle) * speed
        self.balls[0].vy = math.sin(angle) * speed

    def step(self):
        """Advance physics by one timestep."""
        # Move balls
        for b in self.balls:
            if not b.active:
                continue
            b.x += b.vx
            b.y += b.vy

            # Friction
            b.vx *= FRICTION
            b.vy *= FRICTION

            # Stop if very slow
            if abs(b.vx) < MIN_VEL and abs(b.vy) < MIN_VEL:
                b.vx = 0.0
                b.vy = 0.0

            # Wall bounces
            if b.x - BALL_RADIUS < 0:
                b.x = BALL_RADIUS
                b.vx = abs(b.vx) * 0.9
            if b.x + BALL_RADIUS > TABLE_W:
                b.x = TABLE_W - BALL_RADIUS
                b.vx = -abs(b.vx) * 0.9
            if b.y - BALL_RADIUS < 0:
                b.y = BALL_RADIUS
                b.vy = abs(b.vy) * 0.9
            if b.y + BALL_RADIUS > TABLE_H:
                b.y = TABLE_H - BALL_RADIUS
                b.vy = -abs(b.vy) * 0.9

        # Ball-ball collisions (elastic)
        for i in range(len(self.balls)):
            if not self.balls[i].active:
                continue
            for j in range(i + 1, len(self.balls)):
                if not self.balls[j].active:
                    continue
                self._collide(self.balls[i], self.balls[j])

        # Check pockets
        for i, b in enumerate(self.balls):
            if not b.active:
                continue
            for px, py in POCKETS:
                dist = math.sqrt((b.x - px) ** 2 + (b.y - py) ** 2)
                if dist < POCKET_RADIUS:
                    b.active = False
                    b.vx = 0
                    b.vy = 0
                    if i not in self.pocketed:
                        self.pocketed.append(i)

        return self.get_state()

    def _collide(self, a: Ball, b: Ball):
        """Elastic collision between two balls."""
        dx = b.x - a.x
        dy = b.y - a.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < BALL_RADIUS * 2 and dist > 0:
            # Normal vector
            nx = dx / dist
            ny = dy / dist

            # Relative velocity
            dvx = a.vx - b.vx
            dvy = a.vy - b.vy

            # Relative velocity along normal
            dvn = dvx * nx + dvy * ny

            # Only collide if approaching
            if dvn > 0:
                # Equal mass elastic collision
                a.vx -= dvn * nx
                a.vy -= dvn * ny
                b.vx += dvn * nx
                b.vy += dvn * ny

                # Separate balls (prevent overlap)
                overlap = BALL_RADIUS * 2 - dist
                a.x -= overlap * 0.5 * nx
                a.y -= overlap * 0.5 * ny
                b.x += overlap * 0.5 * nx
                b.y += overlap * 0.5 * ny

    def is_settled(self) -> bool:
        """Are all balls stationary?"""
        for b in self.balls:
            if b.active and (abs(b.vx) > MIN_VEL or abs(b.vy) > MIN_VEL):
                return False
        return True

    def get_state(self) -> np.ndarray:
        """State vector: 7 balls × [x, y, vx, vy] = 28 floats.
        Inactive balls get [-1, -1, 0, 0]."""
        state = []
        for b in self.balls:
            if b.active:
                state.extend([b.x / TABLE_W, b.y / TABLE_H,
                              b.vx * 100, b.vy * 100])  # scale velocities
            else:
                state.extend([-1.0, -1.0, 0.0, 0.0])
        return np.array(state, dtype=np.float32)

    def render(self, size: int = 128) -> np.ndarray:
        """Render as (size, size, 3) uint8 RGB image."""
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Scale factors
        sx = size / TABLE_W
        sy = size / TABLE_H
        # Use full image height, adjust for aspect ratio
        actual_h = int(TABLE_H * sx)
        offset_y = (size - actual_h) // 2

        # Border
        img[offset_y:offset_y + 3, :] = BORDER_COLOR
        img[offset_y + actual_h - 3:offset_y + actual_h, :] = BORDER_COLOR
        img[:, :3] = BORDER_COLOR
        img[:, -3:] = BORDER_COLOR

        # Table felt
        img[offset_y + 3:offset_y + actual_h - 3, 3:-3] = TABLE_COLOR

        # Pockets
        for px, py in POCKETS:
            cx = int(px * sx)
            cy = offset_y + int(py * (actual_h / TABLE_H))
            r = int(POCKET_RADIUS * sx)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        ix, iy = cx + dx, cy + dy
                        if 0 <= ix < size and 0 <= iy < size:
                            img[iy, ix] = POCKET_COLOR

        # Balls
        ball_r = max(2, int(BALL_RADIUS * sx))
        for b in self.balls:
            if not b.active:
                continue
            cx = int(b.x * sx)
            cy = offset_y + int(b.y * (actual_h / TABLE_H))
            color = BALL_COLORS[b.color_idx]

            for dy in range(-ball_r, ball_r + 1):
                for dx in range(-ball_r, ball_r + 1):
                    if dx * dx + dy * dy <= ball_r * ball_r:
                        ix, iy = cx + dx, cy + dy
                        if 0 <= ix < size and 0 <= iy < size:
                            # Simple shading
                            shade = 1.0 - (dx * dx + dy * dy) / (ball_r * ball_r) * 0.3
                            img[iy, ix] = tuple(int(c * shade) for c in color)

        return img


def generate_dataset(n_episodes=1000, steps_per_ep=60, frameskip=5, seed=42):
    """Generate pool training data: episodes of shots with multi-body collisions."""
    env = PoolWorld()
    rng = np.random.default_rng(seed)

    all_frames, all_states, all_actions, all_episodes = [], [], [], []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)

        # Random shot
        angle = rng.uniform(0, 2 * math.pi)
        power = rng.uniform(0.3, 1.0)

        env.shoot(angle, power)
        action = np.array([angle / (2 * math.pi), power], dtype=np.float32)

        for step in range(steps_per_ep * frameskip):
            env.step()

            if step % frameskip == 0:
                frame = env.render(128)
                state = env.get_state()
                # Action: only active during first few training steps
                step_action = action if (step // frameskip) < 4 else np.zeros(2, dtype=np.float32)

                all_frames.append(frame)
                all_states.append(state)
                all_actions.append(step_action)
                all_episodes.append(ep)

        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"  Episode {ep + 1}/{n_episodes} "
                  f"(angle={angle:.2f}, power={power:.2f}, "
                  f"pocketed={len(env.pocketed)})")

    return {
        "frames": np.array(all_frames),     # (N, 128, 128, 3) uint8
        "states": np.array(all_states),      # (N, 28) float32
        "actions": np.array(all_actions),    # (N, 2) float32
        "episodes": np.array(all_episodes),  # (N,) int
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes", type=int, default=1000)
    pa.add_argument("--steps", type=int, default=60)
    pa.add_argument("--output", default="data/pool_v1.npz")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--preview", action="store_true")
    args = pa.parse_args()

    if args.preview:
        from PIL import Image

        env = PoolWorld()
        env.reset(seed=42)
        img = env.render(256)
        Image.fromarray(img).save("/tmp/pool_start.png")
        print("Saved /tmp/pool_start.png")

        env.shoot(angle=0.3, power=0.8)
        for i in range(150):
            env.step()
            if i % 30 == 0:
                img = env.render(256)
                Image.fromarray(img).save(f"/tmp/pool_step_{i}.png")
                print(f"Saved /tmp/pool_step_{i}.png (settled={env.is_settled()})")
    else:
        print(f"Generating {args.episodes} pool episodes × {args.steps} steps "
              f"(frameskip=5)...")
        data = generate_dataset(args.episodes, args.steps, seed=args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **data)
        print(f"Saved {len(data['frames'])} frames to {args.output}")
        for k in data:
            print(f"  {k}: {data[k].shape}")
