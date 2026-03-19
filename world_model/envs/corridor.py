"""Procedural alien corridor environment for AURA.

A Gymnasium environment that renders 64x64 RGB egocentric views of
procedurally generated corridors using a vectorized NumPy raycaster.
Audio context vector drives visual properties.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from world_model.audio.features import unpack_context

# Color palette constants
_COOL = np.array([40, 20, 120], dtype=np.float32)
_WARM = np.array([180, 60, 30], dtype=np.float32)
_GLOW = np.array([0, 40, 60], dtype=np.float32)

FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2


class CorridorEnv(gym.Env):
    """Alien corridor with audio-conditioned visuals.

    Observations:
        image: (64, 64, 3) uint8 RGB
        context: (16,) float32 audio context in [0, 1]

    Actions:
        0: move forward
        1: turn left
        2: turn right
    """

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, render_mode: str = 'rgb_array', map_size: int = 32,
                 view_size: int = 64, fov: float = 60.0, max_steps: int = 500):
        super().__init__()
        self.render_mode = render_mode
        self.map_size = map_size
        self.view_size = view_size
        self.fov = np.radians(fov)
        self.max_steps = max_steps

        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (view_size, view_size, 3), dtype=np.uint8),
            'context': spaces.Box(0.0, 1.0, (16,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(3)

        self._map = None
        self._pos = None
        self._angle = None
        self._context = np.zeros(16, dtype=np.float32)
        self._step_count = 0
        self._rng = np.random.default_rng()

    def set_context(self, context: np.ndarray):
        """Set the audio context vector for visual conditioning."""
        self._context = np.clip(context, 0.0, 1.0).astype(np.float32)

    def _generate_map(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a corridor map via random walk with branching."""
        grid = np.ones((self.map_size, self.map_size), dtype=np.uint8)
        cx, cy = self.map_size // 2, self.map_size // 2
        pos = [cx, cy]
        grid[pos[0], pos[1]] = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for _ in range(self.map_size * self.map_size // 3):
            dx, dy = directions[rng.integers(4)]
            for _ in range(rng.integers(2, 5)):
                nx, ny = pos[0] + dx, pos[1] + dy
                if 1 <= nx < self.map_size - 1 and 1 <= ny < self.map_size - 1:
                    pos = [nx, ny]
                    grid[nx, ny] = 0
                    if rng.random() > 0.6:
                        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            wx, wy = nx + ddx, ny + ddy
                            if 1 <= wx < self.map_size - 1 and 1 <= wy < self.map_size - 1:
                                grid[wx, wy] = 0
            if rng.random() > 0.7:
                pos = [cx, cy]

        return grid

    def _find_spawn(self, rng: np.random.Generator) -> tuple[float, float]:
        floors = np.argwhere(self._map == 0)
        idx = rng.integers(len(floors))
        return float(floors[idx][0]) + 0.5, float(floors[idx][1]) + 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._map = self._generate_map(self._rng)
        px, py = self._find_spawn(self._rng)
        self._pos = np.array([px, py], dtype=np.float64)
        self._angle = self._rng.uniform(0, 2 * np.pi)
        self._step_count = 0
        self._context = np.zeros(16, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: int):
        self._step_count += 1
        move_speed = 0.15
        turn_speed = np.radians(15)

        if action == FORWARD:
            dx = np.cos(self._angle) * move_speed
            dy = np.sin(self._angle) * move_speed
            new_pos = self._pos + np.array([dx, dy])
            gx, gy = int(new_pos[0]), int(new_pos[1])
            if (0 <= gx < self.map_size and 0 <= gy < self.map_size
                    and self._map[gx, gy] == 0):
                self._pos = new_pos
        elif action == TURN_LEFT:
            self._angle -= turn_speed
        elif action == TURN_RIGHT:
            self._angle += turn_speed

        self._angle %= 2 * np.pi
        reward = -0.01 + (0.02 if action == FORWARD else 0.0)
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, False, truncated, {}

    def _get_obs(self) -> dict:
        return {
            'image': self._render_frame(),
            'context': self._context.copy(),
        }

    def _render_frame(self) -> np.ndarray:
        """Render a 64x64 RGB frame using vectorized raycasting."""
        w = h = self.view_size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        px, py = self._pos
        af = unpack_context(self._context)

        # --- Vectorized DDA: cast all W rays at once ---
        xs = np.arange(w, dtype=np.float64)
        ray_offsets = (xs / w - 0.5) * self.fov
        ray_angles = self._angle + ray_offsets
        ray_dx = np.cos(ray_angles)
        ray_dy = np.sin(ray_angles)

        # Avoid division by zero
        safe_dx = np.where(ray_dx == 0, 1e-10, ray_dx)
        safe_dy = np.where(ray_dy == 0, 1e-10, ray_dy)
        delta_x = np.abs(1.0 / safe_dx)
        delta_y = np.abs(1.0 / safe_dy)

        map_x = np.full(w, int(px), dtype=np.int32)
        map_y = np.full(w, int(py), dtype=np.int32)

        step_x = np.where(ray_dx < 0, -1, 1).astype(np.int32)
        step_y = np.where(ray_dy < 0, -1, 1).astype(np.int32)

        side_dist_x = np.where(ray_dx < 0, (px - map_x) * delta_x, (map_x + 1.0 - px) * delta_x)
        side_dist_y = np.where(ray_dy < 0, (py - map_y) * delta_y, (map_y + 1.0 - py) * delta_y)

        hit = np.zeros(w, dtype=bool)
        side = np.zeros(w, dtype=np.int32)
        active = np.ones(w, dtype=bool)

        for _ in range(64):
            if not active.any():
                break
            go_x = active & (side_dist_x < side_dist_y)
            go_y = active & ~go_x

            side_dist_x = np.where(go_x, side_dist_x + delta_x, side_dist_x)
            map_x = np.where(go_x, map_x + step_x, map_x)
            side = np.where(go_x, 0, side)

            side_dist_y = np.where(go_y, side_dist_y + delta_y, side_dist_y)
            map_y = np.where(go_y, map_y + step_y, map_y)
            side = np.where(go_y, 1, side)

            oob = active & ((map_x < 0) | (map_x >= self.map_size) |
                            (map_y < 0) | (map_y >= self.map_size))
            active &= ~oob

            # Safe lookup (clamp OOB to 0,0 which is always wall)
            safe_mx = np.clip(map_x, 0, self.map_size - 1)
            safe_my = np.clip(map_y, 0, self.map_size - 1)
            wall_hit = active & (self._map[safe_mx, safe_my] == 1)
            hit |= wall_hit
            active &= ~wall_hit

        # Perpendicular distance
        perp_dist = np.where(side == 0, side_dist_x - delta_x, side_dist_y - delta_y)
        perp_dist = np.maximum(perp_dist, 0.01)

        # Wall geometry
        line_height = (h / perp_dist).astype(np.int32)
        draw_start = np.clip(h // 2 - line_height // 2, 0, h)
        draw_end = np.clip(h // 2 + line_height // 2, 0, h)

        # --- Compute wall colors (vectorized over columns) ---
        base = _COOL * (1 - af['temperature']) + _WARM * af['temperature']  # (3,)
        gray = np.mean(base)
        base = gray + (base - gray) * (0.3 + af['mid'] * 0.7)
        base = base + _GLOW * af['bass']

        brightness = 0.3 + af['rms'] * 0.7
        fog = np.maximum(0.1, 1.0 - perp_dist / 12.0)  # (W,)
        col_brightness = brightness * fog  # (W,)
        side_shade = np.where(side == 1, 0.7, 1.0)  # (W,)

        noise = 1.0 + af['high'] * 0.2 * (self._rng.random(w) - 0.5)
        wall_colors = (base[None, :] * (col_brightness * side_shade * noise)[:, None])
        wall_colors = np.clip(wall_colors, 0, 255).astype(np.uint8)  # (W, 3)

        # --- Compute floor/ceiling colors (vectorized) ---
        floor_base = np.array([20 + af['temperature'] * 30, 15,
                               25 + (1 - af['temperature']) * 20], dtype=np.float32)
        ceil_base = np.array([10, 8 + af['temperature'] * 15,
                              15 + (1 - af['temperature']) * 15], dtype=np.float32)

        # --- Fill image column by column (only the slice logic, colors are precomputed) ---
        ys = np.arange(h, dtype=np.float32)
        for x in range(w):
            if not hit[x]:
                # No wall — all ceiling
                fracs = ys / h
                colors = ceil_base[None, :] * ((0.2 + af['rms'] * 0.3) * (0.5 + (1 - fracs) * 0.5))[:, None]
                image[:, x] = np.clip(colors, 0, 255).astype(np.uint8)
                continue

            ds, de = draw_start[x], draw_end[x]

            # Wall
            image[ds:de, x] = wall_colors[x]

            # Ceiling
            if ds > 0:
                fracs = ys[:ds] / max(ds, 1)
                colors = ceil_base[None, :] * ((0.2 + af['rms'] * 0.3) * (0.5 + (1 - fracs) * 0.5))[:, None]
                image[:ds, x] = np.clip(colors, 0, 255).astype(np.uint8)

            # Floor
            if de < h:
                n_floor = h - de
                fracs = (ys[de:] - de) / max(n_floor, 1)
                colors = floor_base[None, :] * ((0.3 + af['rms'] * 0.4) * (0.5 + fracs * 0.5))[:, None]
                image[de:, x] = np.clip(colors, 0, 255).astype(np.uint8)

        return image

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        return None
