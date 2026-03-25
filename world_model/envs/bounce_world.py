"""Audio-reactive bouncing ball physics environment.

Simple 2D physics: ball with gravity, bouncing off walls and floor.
Audio controls the physics:
  - RMS (volume) = launch force (scream to launch!)
  - Onset (clap) = instant upward kick
  - Bass = gravity strength
  - Mid = horizontal wind
  - High = ball size / trail length
  - Temperature = color palette (for pixel mode)

Renders as ASCII art (40x80) or pixel (64x64).

Usage:
    env = BounceWorld()
    env.reset()
    for audio in audio_stream:
        env.step(audio)
        frame = env.render_ascii()  # 40x80 string
"""
from __future__ import annotations
import numpy as np


# ASCII glyphs for the ball at different sizes
BALL_GLYPHS = list("·•●◉⬤")
TRAIL_GLYPHS = list("·∙°˚")
FLOOR_CHAR = "═"
WALL_CHAR = "║"
CORNER_CHARS = ("╔", "╗", "╚", "╝")
BG_CHAR = " "

# Particle burst on impact
BURST_CHARS = list("*✦✧★☆⚡✹·")


class BounceWorld:
    """2D bouncing ball with audio-reactive physics."""

    def __init__(self, width: int = 80, height: int = 40):
        self.W = width
        self.H = height
        # Physics state
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.gravity = 0.5
        self.bounce = 0.7  # energy retention on bounce
        self.trail: list[tuple[float, float]] = []
        self.particles: list[dict] = []
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.ball_x = self.W / 2
        self.ball_y = self.H / 2
        self.vel_x = self.rng.uniform(-2, 2)
        self.vel_y = 0.0
        self.trail = []
        self.particles = []
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return state vector: [ball_x, ball_y, vel_x, vel_y, gravity]."""
        return np.array([
            self.ball_x / self.W,   # normalized 0-1
            self.ball_y / self.H,
            self.vel_x / 10.0,      # normalized roughly
            self.vel_y / 10.0,
            self.gravity / 2.0,
        ], dtype=np.float32)

    def step(self, audio: np.ndarray) -> np.ndarray:
        """Advance physics by one frame. Audio is 16-float context vector.

        Returns state vector (5 floats).
        """
        # Unpack audio
        bass = (audio[0] + audio[1]) / 2
        mid = (audio[2] + audio[3]) / 2
        high = (audio[4] + audio[5]) / 2
        onset = (audio[6] + audio[7]) / 2
        bpm = (audio[8] + audio[9]) / 2
        temp = (audio[10] + audio[11]) / 2
        rms = (audio[12] + audio[13]) / 2

        # Audio affects physics
        self.gravity = 0.3 + bass * 1.2        # bass = heavier gravity
        wind = (mid - 0.5) * 2.0               # mid = horizontal wind
        self.vel_x += wind * 0.3

        # RMS (volume) = upward force — SCREAM TO LAUNCH
        if rms > 0.3:
            launch_force = (rms - 0.3) * 15.0
            self.vel_y -= launch_force * 0.3   # up is negative y

        # Onset (clap/beat) = instant kick
        if onset > 0.4:
            kick = onset * 8.0
            self.vel_y -= kick
            # Spawn burst particles
            for _ in range(int(onset * 10)):
                self.particles.append({
                    'x': self.ball_x, 'y': self.ball_y,
                    'vx': self.rng.uniform(-3, 3),
                    'vy': self.rng.uniform(-5, 1),
                    'life': self.rng.integers(3, 10),
                })

        # Apply gravity
        self.vel_y += self.gravity

        # Update position
        self.ball_x += self.vel_x
        self.ball_y += self.vel_y

        # Trail
        self.trail.append((self.ball_x, self.ball_y))
        max_trail = 5 + int(high * 15)  # high freq = longer trail
        if len(self.trail) > max_trail:
            self.trail = self.trail[-max_trail:]

        # Floor bounce
        floor = self.H - 2
        if self.ball_y >= floor:
            self.ball_y = floor
            self.vel_y = -abs(self.vel_y) * self.bounce
            # Spawn floor particles
            if abs(self.vel_y) > 1:
                for _ in range(min(8, int(abs(self.vel_y) * 2))):
                    self.particles.append({
                        'x': self.ball_x + self.rng.uniform(-2, 2),
                        'y': floor,
                        'vx': self.rng.uniform(-2, 2),
                        'vy': self.rng.uniform(-3, 0),
                        'life': self.rng.integers(2, 6),
                    })
            # Dampen small bounces
            if abs(self.vel_y) < 0.5:
                self.vel_y = 0

        # Ceiling bounce
        if self.ball_y < 1:
            self.ball_y = 1
            self.vel_y = abs(self.vel_y) * self.bounce

        # Wall bounce
        if self.ball_x < 1:
            self.ball_x = 1
            self.vel_x = abs(self.vel_x) * self.bounce
        if self.ball_x >= self.W - 1:
            self.ball_x = self.W - 2
            self.vel_x = -abs(self.vel_x) * self.bounce

        # Friction
        self.vel_x *= 0.98

        # Update particles
        alive = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.3  # particle gravity
            p['life'] -= 1
            if p['life'] > 0 and 0 <= p['x'] < self.W and 0 <= p['y'] < self.H:
                alive.append(p)
        self.particles = alive

        return self.get_state()

    def render_ascii(self, audio: np.ndarray = None) -> str:
        """Render current state as 40x80 ASCII art."""
        grid = [[BG_CHAR] * self.W for _ in range(self.H)]

        # Floor
        for x in range(self.W):
            grid[self.H - 1][x] = FLOOR_CHAR

        # Walls
        for y in range(self.H):
            grid[y][0] = WALL_CHAR
            grid[y][self.W - 1] = WALL_CHAR

        # Corners
        grid[0][0] = CORNER_CHARS[0]
        grid[0][self.W - 1] = CORNER_CHARS[1]
        grid[self.H - 1][0] = CORNER_CHARS[2]
        grid[self.H - 1][self.W - 1] = CORNER_CHARS[3]

        # Top wall
        for x in range(1, self.W - 1):
            grid[0][x] = FLOOR_CHAR

        # Trail
        for i, (tx, ty) in enumerate(self.trail):
            ix, iy = int(tx), int(ty)
            if 1 <= ix < self.W - 1 and 1 <= iy < self.H - 1:
                t_idx = min(len(TRAIL_GLYPHS) - 1,
                            int((1 - i / max(len(self.trail), 1)) * len(TRAIL_GLYPHS)))
                grid[iy][ix] = TRAIL_GLYPHS[t_idx]

        # Particles
        for p in self.particles:
            px, py = int(p['x']), int(p['y'])
            if 1 <= px < self.W - 1 and 1 <= py < self.H - 1:
                grid[py][px] = self.rng.choice(BURST_CHARS)

        # Ball
        bx, by = int(self.ball_x), int(self.ball_y)
        # Ball size based on audio high freq
        ball_size = 1
        if audio is not None:
            high = (audio[4] + audio[5]) / 2
            rms = (audio[12] + audio[13]) / 2
            ball_size = 1 + int(rms * 3)
            ball_glyph_idx = min(len(BALL_GLYPHS) - 1, int(rms * len(BALL_GLYPHS)))
        else:
            ball_glyph_idx = 2

        ball_char = BALL_GLYPHS[ball_glyph_idx]
        for dy in range(-ball_size + 1, ball_size):
            for dx in range(-ball_size + 1, ball_size):
                px, py = bx + dx, by + dy
                if 1 <= px < self.W - 1 and 1 <= py < self.H - 1:
                    if dx * dx + dy * dy < ball_size * ball_size:
                        grid[py][px] = ball_char

        # Audio meter bar at top
        if audio is not None:
            rms = (audio[12] + audio[13]) / 2
            bar_len = int(rms * (self.W - 4))
            for x in range(2, 2 + bar_len):
                if x < self.W - 1:
                    grid[0][x] = "▓" if rms > 0.6 else "▒" if rms > 0.3 else "░"

        return "\n".join("".join(row) for row in grid)

    def render_pixel(self, audio: np.ndarray = None, size: int = 64) -> np.ndarray:
        """Render as (size, size, 3) uint8 RGB image."""
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Scale positions
        sx = size / self.W
        sy = size / self.H

        # Background color from temperature
        temp = 0.5
        if audio is not None:
            temp = (audio[10] + audio[11]) / 2

        # Floor
        floor_y = int((self.H - 1) * sy)
        img[floor_y:, :] = [30, 30, 50]

        # Trail
        for i, (tx, ty) in enumerate(self.trail):
            px = int(tx * sx)
            py = int(ty * sy)
            alpha = i / max(len(self.trail), 1)
            c = int(40 + alpha * 60)
            if 0 <= px < size and 0 <= py < size:
                img[py, px] = [c, c // 2, c]

        # Ball
        bx = int(self.ball_x * sx)
        by = int(self.ball_y * sy)
        r = max(1, int(2 * sx))
        if audio is not None:
            rms = (audio[12] + audio[13]) / 2
            r = max(1, int((1 + rms * 3) * sx))

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = bx + dx, by + dy
                    if 0 <= px < size and 0 <= py < size:
                        # Color based on temperature
                        if temp > 0.5:
                            img[py, px] = [255, 140, 50]  # warm
                        else:
                            img[py, px] = [50, 200, 255]  # cool

        # Particles
        for p in self.particles:
            px = int(p['x'] * sx)
            py = int(p['y'] * sy)
            if 0 <= px < size and 0 <= py < size:
                img[py, px] = [255, 255, 200]

        return img


def generate_dataset(n_episodes=200, steps=200, seed=42):
    """Generate training data: (frames, states, audios)."""
    from world_model.ascii_model.model import frame_to_indices

    env = BounceWorld()
    rng = np.random.default_rng(seed)

    all_frames, all_states, all_audios, all_episodes = [], [], [], []
    profiles = ['high', 'low', 'ramp', 'pulse', 'random', 'sweep']

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        profile = profiles[ep % len(profiles)]

        for step_i in range(steps):
            t = step_i / steps
            ctx = np.zeros(16, dtype=np.float32)

            if profile == 'high':
                for i in range(0, 14, 2):
                    ctx[i] = ctx[i+1] = rng.uniform(0.6, 0.95)
                if rng.random() < 0.15:
                    ctx[6] = ctx[7] = rng.uniform(0.6, 1.0)
            elif profile == 'low':
                for i in range(0, 14, 2):
                    ctx[i] = ctx[i+1] = rng.uniform(0.02, 0.15)
            elif profile == 'ramp':
                for i in range(0, 14, 2):
                    ctx[i] = ctx[i+1] = t * rng.uniform(0.8, 1.0)
                if t > 0.5 and rng.random() < 0.1:
                    ctx[6] = ctx[7] = rng.uniform(0.5, 1.0)
            elif profile == 'pulse':
                val = 0.8 if (step_i // 15) % 2 == 0 else 0.1
                for i in range(0, 14, 2):
                    ctx[i] = ctx[i+1] = val + rng.uniform(-0.05, 0.05)
                if val > 0.5:
                    ctx[6] = ctx[7] = 0.7
            elif profile == 'random':
                for i in range(0, 14, 2):
                    ctx[i] = ctx[i+1] = rng.uniform(0, 1)
                if rng.random() < 0.15:
                    ctx[6] = ctx[7] = rng.uniform(0.5, 1.0)
            elif profile == 'sweep':
                ctx[0] = ctx[1] = 1.0 - t
                ctx[2] = ctx[3] = 0.5
                ctx[4] = ctx[5] = t
                ctx[10] = ctx[11] = t
                ctx[12] = ctx[13] = 0.3 + 0.5 * np.sin(2 * np.pi * t * 3)

            ctx = np.clip(ctx, 0, 1)
            state = env.step(ctx)
            frame_str = env.render_ascii(ctx)
            frame_idx = frame_to_indices(frame_str)

            all_frames.append(frame_idx)
            all_states.append(state)
            all_audios.append(ctx)
            all_episodes.append(ep)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes} ({profile})")

    return {
        "frames": np.array(all_frames),
        "states": np.array(all_states),
        "audios": np.array(all_audios),
        "episodes": np.array(all_episodes),
    }


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes", type=int, default=200)
    pa.add_argument("--steps", type=int, default=200)
    pa.add_argument("--output", default="data/bounce_v1.npz")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--preview", action="store_true", help="Show a few frames")
    args = pa.parse_args()

    if args.preview:
        env = BounceWorld()
        env.reset(seed=42)
        ctx = np.zeros(16, dtype=np.float32)
        ctx[12] = ctx[13] = 0.8  # high volume
        ctx[6] = ctx[7] = 0.9    # onset
        for i in range(5):
            env.step(ctx)
        # Low volume
        ctx[12] = ctx[13] = 0.1
        ctx[6] = ctx[7] = 0.0
        for i in range(20):
            env.step(ctx)
        print(env.render_ascii(ctx))
    else:
        print(f"Generating {args.episodes} episodes x {args.steps} steps...")
        data = generate_dataset(args.episodes, args.steps, args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **data)
        print(f"Saved {len(data['frames'])} frames to {args.output}")
        for k in data:
            print(f"  {k}: {data[k].shape}")
