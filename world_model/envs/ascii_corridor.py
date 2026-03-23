"""ASCII/Unicode raycaster renderer for AURA corridors.

Renders the same procedural alien corridor as corridor.py, but outputs
Unicode glyph grids instead of RGB images.  Audio context drives glyph
selection: bass shifts wall density, RMS modulates intensity, onsets
inject flash characters, spectral centroid shifts between cool/warm
box-drawing palettes.

Usage:
    # Gymnasium env (returns string observations)
    env = AsciiCorridorEnv(cols=80, rows=40)
    obs, info = env.reset()
    print(obs['ascii'])

    # CLI live mode (curses)
    python -m world_model.envs.ascii_corridor --live

    # Generate JSONL dataset
    python -m world_model.envs.ascii_corridor --generate --episodes 5 \
        --steps 100 --output frames.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

def unpack_context(c: np.ndarray) -> dict[str, float]:
    """Unpack 16-float audio context vector into named features."""
    return {
        'bass': float((c[0] + c[1]) / 2),
        'mid': float((c[2] + c[3]) / 2),
        'high': float((c[4] + c[5]) / 2),
        'onset': float((c[6] + c[7]) / 2),
        'bpm': float((c[8] + c[9]) / 2),
        'temperature': float((c[10] + c[11]) / 2),
        'rms': float((c[12] + c[13]) / 2),
    }

# ---------------------------------------------------------------------------
# Glyph palettes — ordered near-to-far (index 0 = closest)
# ---------------------------------------------------------------------------

# Walls: dense to sparse shading blocks
WALL_GLYPHS_COOL = list("█▓▒░╔═╗║")
WALL_GLYPHS_WARM = list("█▓▒░╭─╮│")

# Floor: near to far
FLOOR_GLYPHS = list("·.:; ")

# Ceiling: near to far
CEIL_GLYPHS = list("`'\"* ")

# Onset flash characters (randomly sampled when onset > threshold)
FLASH_CHARS = list("✦✧★☆⚡✹✸")

# High-intensity wall accents driven by RMS
INTENSE_WALL = list("▉▊▋▌")

FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2


# ---------------------------------------------------------------------------
# Pure-function ASCII renderer: distances + audio -> glyph grid
# ---------------------------------------------------------------------------

def render_ascii(
    perp_dist: np.ndarray,
    hit: np.ndarray,
    side: np.ndarray,
    context: np.ndarray,
    cols: int = 80,
    rows: int = 40,
    max_dist: float = 20.0,
    rng: np.random.Generator | None = None,
) -> str:
    """Convert raycasted distances into a Unicode glyph grid.

    Parameters
    ----------
    perp_dist : (cols,) float  — perpendicular wall distance per column
    hit       : (cols,) bool   — whether a wall was hit
    side      : (cols,) int    — 0=X-side, 1=Y-side
    context   : (16,) float32  — audio context vector
    cols, rows: grid dimensions
    max_dist  : distance at which fog fully obscures
    rng       : random generator for stochastic glyphs

    Returns
    -------
    A single string of *rows* lines, each *cols* characters wide, joined
    by newlines.  Total length = cols*rows + (rows-1).
    """
    if rng is None:
        rng = np.random.default_rng()

    af = unpack_context(context)
    bass = af['bass']
    mid = af['mid']
    high = af['high']
    onset = af['onset']
    rms = af['rms']
    temperature = af['temperature']

    # Pick wall palette based on temperature (cool vs warm box-drawing)
    # Blend: temperature < 0.5 → cool, >= 0.5 → warm
    wall_glyphs = WALL_GLYPHS_WARM if temperature >= 0.5 else WALL_GLYPHS_COOL
    n_wall = len(wall_glyphs)
    n_floor = len(FLOOR_GLYPHS)
    n_ceil = len(CEIL_GLYPHS)

    # Bass increases perceived wall density (shifts glyph index toward 0 = dense)
    bass_shift = -bass * 1.5  # negative = shift toward denser glyphs

    # RMS drives intensity: at high RMS, inject INTENSE_WALL glyphs
    rms_threshold = 0.6

    # Onset flash probability per character
    onset_flash_prob = onset * 0.4  # up to 40% of chars flash at max onset

    # --- Geometry: same as corridor.py RGB renderer ---
    line_height = (rows / np.maximum(perp_dist, 0.01)).astype(np.int32)
    draw_start = np.clip(rows // 2 - line_height // 2, 0, rows)
    draw_end = np.clip(rows // 2 + line_height // 2, 0, rows)

    # Fog factor per column
    fog = np.clip(1.0 - perp_dist / max_dist, 0.0, 1.0)

    # Build the grid row by row
    grid = []
    for row in range(rows):
        line_chars = []
        for col in range(cols):
            if not hit[col]:
                # No wall hit — deep fog / void
                line_chars.append(' ')
                continue

            ds = draw_start[col]
            de = draw_end[col]

            # ----- Onset flash override (sparse random) -----
            if onset_flash_prob > 0 and rng.random() < onset_flash_prob:
                line_chars.append(rng.choice(FLASH_CHARS))
                continue

            if row < ds:
                # ---- Ceiling ----
                if ds == 0:
                    frac = 0.0
                else:
                    frac = 1.0 - (row / ds)  # 0 at wall edge, 1 at top
                idx = int(frac * (n_ceil - 1))
                # High frequency adds detail / shifts to denser ceiling
                idx = max(0, idx - int(high * 1.5))
                idx = np.clip(idx, 0, n_ceil - 1)
                ch = CEIL_GLYPHS[idx]
            elif row >= de:
                # ---- Floor ----
                if de >= rows:
                    frac = 0.0
                else:
                    frac = (row - de) / max(rows - de, 1)  # 0 at wall base, 1 at bottom
                idx = int(frac * (n_floor - 1))
                # Mid energy adds floor detail
                idx = max(0, idx - int(mid * 1.0))
                idx = np.clip(idx, 0, n_floor - 1)
                ch = FLOOR_GLYPHS[idx]
            else:
                # ---- Wall ----
                dist_norm = np.clip(perp_dist[col] / max_dist, 0.0, 1.0)
                # Base glyph from distance (0=near/dense, n_wall-1=far/sparse)
                idx_f = dist_norm * (n_wall - 1) + bass_shift
                idx = int(np.clip(idx_f, 0, n_wall - 1))

                # Side shading: Y-side walls one step lighter
                if side[col] == 1:
                    idx = min(idx + 1, n_wall - 1)

                # RMS intensity override: very close + high RMS → intense glyph
                if rms > rms_threshold and dist_norm < 0.3:
                    intense_idx = int((1.0 - dist_norm / 0.3) * (len(INTENSE_WALL) - 1))
                    intense_idx = np.clip(intense_idx, 0, len(INTENSE_WALL) - 1)
                    ch = INTENSE_WALL[intense_idx]
                else:
                    ch = wall_glyphs[idx]

            # Fog: far-away characters fade to space
            if fog[col] < 0.15:
                ch = ' '
            elif fog[col] < 0.3:
                # Very faint — use the sparsest glyph category
                if row < ds:
                    ch = CEIL_GLYPHS[-2] if n_ceil > 1 else ' '
                elif row >= de:
                    ch = FLOOR_GLYPHS[-2] if n_floor > 1 else ' '
                # walls keep their glyph (already sparse at distance)

            line_chars.append(ch)
        grid.append(''.join(line_chars))

    return '\n'.join(grid)


def format_context_line(context: np.ndarray) -> str:
    """One-line summary of audio context for display above the frame."""
    af = unpack_context(context)
    return (
        f"[b={af['bass']:.2f} m={af['mid']:.2f} h={af['high']:.2f} "
        f"o={af['onset']:.2f} bpm={af['bpm']:.2f} "
        f"t={af['temperature']:.2f} rms={af['rms']:.2f}]"
    )


# ---------------------------------------------------------------------------
# Gymnasium environment wrapper
# ---------------------------------------------------------------------------

class AsciiCorridorEnv(gym.Env):
    """Alien corridor with ASCII/Unicode rendering.

    Wraps the same map generation, raycasting, and physics as CorridorEnv
    but produces text output instead of RGB images.

    Observations:
        ascii  : str  — the rendered frame (cols x rows grid)
        context: (16,) float32 audio context
        image  : (rows, cols) uint8 — optional numeric glyph-index matrix

    Actions:
        0: move forward
        1: turn left
        2: turn right
    """

    metadata = {'render_modes': ['ansi', 'rgb_array']}

    def __init__(
        self,
        render_mode: str = 'ansi',
        map_size: int = 32,
        cols: int = 80,
        rows: int = 40,
        fov: float = 60.0,
        max_steps: int = 500,
        max_dist: float = 20.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.map_size = map_size
        self.cols = cols
        self.rows = rows
        self.fov = np.radians(fov)
        self.max_steps = max_steps
        self.max_dist = max_dist

        self.observation_space = spaces.Dict({
            'ascii': spaces.Text(
                min_length=0,
                max_length=cols * rows + rows,
            ),
            'context': spaces.Box(0.0, 1.0, (16,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(3)

        self._map: np.ndarray | None = None
        self._pos: np.ndarray | None = None
        self._angle: float = 0.0
        self._context = np.zeros(16, dtype=np.float32)
        self._step_count = 0
        self._rng = np.random.default_rng()

        # Cache last raycast for render()
        self._last_perp_dist: np.ndarray | None = None
        self._last_hit: np.ndarray | None = None
        self._last_side: np.ndarray | None = None

    # ----- audio context -----

    def set_context(self, context: np.ndarray):
        """Set the 16-float audio context vector."""
        self._context = np.clip(context, 0.0, 1.0).astype(np.float32)

    # ----- map generation (identical to CorridorEnv) -----

    def _generate_map(self, rng: np.random.Generator) -> np.ndarray:
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

    # ----- gymnasium interface -----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._map = self._generate_map(self._rng)
        px, py = self._find_spawn(self._rng)
        self._pos = np.array([px, py], dtype=np.float64)
        self._angle = self._rng.uniform(0, 2 * np.pi)
        self._step_count = 0
        self._context = np.zeros(16, dtype=np.float32)
        self._raycast()
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
        self._raycast()
        reward = -0.01 + (0.02 if action == FORWARD else 0.0)
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, False, truncated, {}

    # ----- raycasting (vectorized DDA, same algorithm as CorridorEnv) -----

    def _raycast(self):
        """Run the DDA raycaster and cache per-column distances."""
        w = self.cols
        px, py = self._pos

        xs = np.arange(w, dtype=np.float64)
        ray_offsets = (xs / w - 0.5) * self.fov
        ray_angles = self._angle + ray_offsets
        ray_dx = np.cos(ray_angles)
        ray_dy = np.sin(ray_angles)

        safe_dx = np.where(ray_dx == 0, 1e-10, ray_dx)
        safe_dy = np.where(ray_dy == 0, 1e-10, ray_dy)
        delta_x = np.abs(1.0 / safe_dx)
        delta_y = np.abs(1.0 / safe_dy)

        map_x = np.full(w, int(px), dtype=np.int32)
        map_y = np.full(w, int(py), dtype=np.int32)

        step_x = np.where(ray_dx < 0, -1, 1).astype(np.int32)
        step_y = np.where(ray_dy < 0, -1, 1).astype(np.int32)

        side_dist_x = np.where(
            ray_dx < 0, (px - map_x) * delta_x, (map_x + 1.0 - px) * delta_x
        )
        side_dist_y = np.where(
            ray_dy < 0, (py - map_y) * delta_y, (map_y + 1.0 - py) * delta_y
        )

        hit = np.zeros(w, dtype=bool)
        side = np.zeros(w, dtype=np.int32)
        active = np.ones(w, dtype=bool)

        max_steps = max(self.map_size * 2, 64)
        for _ in range(max_steps):
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

            oob = active & (
                (map_x < 0) | (map_x >= self.map_size)
                | (map_y < 0) | (map_y >= self.map_size)
            )
            active &= ~oob

            safe_mx = np.clip(map_x, 0, self.map_size - 1)
            safe_my = np.clip(map_y, 0, self.map_size - 1)
            wall_hit = active & (self._map[safe_mx, safe_my] == 1)
            hit |= wall_hit
            active &= ~wall_hit

        perp_dist = np.where(side == 0, side_dist_x - delta_x, side_dist_y - delta_y)
        perp_dist = np.maximum(perp_dist, 0.01)

        self._last_perp_dist = perp_dist
        self._last_hit = hit
        self._last_side = side

    # ----- observation / render -----

    def _get_obs(self) -> dict:
        ascii_frame = render_ascii(
            self._last_perp_dist,
            self._last_hit,
            self._last_side,
            self._context,
            cols=self.cols,
            rows=self.rows,
            max_dist=self.max_dist,
            rng=self._rng,
        )
        return {
            'ascii': ascii_frame,
            'context': self._context.copy(),
        }

    def render(self):
        if self.render_mode == 'ansi':
            obs = self._get_obs()
            header = format_context_line(self._context)
            return header + '\n' + obs['ascii']
        return None


# ---------------------------------------------------------------------------
# CLI: live terminal mode + JSONL generation
# ---------------------------------------------------------------------------

def _run_live(args):
    """Render frames live in the terminal."""
    env = AsciiCorridorEnv(
        cols=args.cols, rows=args.rows, map_size=args.map_size,
        max_steps=args.steps, render_mode='ansi',
    )

    # If an audio context is provided, use it
    ctx = np.zeros(16, dtype=np.float32)
    if args.bass is not None:
        ctx[0] = ctx[1] = args.bass
    if args.rms is not None:
        ctx[12] = ctx[13] = args.rms
    if args.temperature is not None:
        ctx[10] = ctx[11] = args.temperature

    obs, _ = env.reset(seed=args.seed)
    env.set_context(ctx)

    use_curses = False
    if args.curses:
        try:
            import curses as _curses
            use_curses = True
        except ImportError:
            pass

    if use_curses:
        _run_live_curses(env, ctx, args)
    else:
        _run_live_print(env, ctx, args)


def _run_live_print(env, ctx, args):
    """Simple print-and-clear terminal mode."""
    rng = np.random.default_rng(args.seed)
    actions = [FORWARD, FORWARD, FORWARD, TURN_LEFT, TURN_RIGHT]

    try:
        for step_i in range(args.steps):
            # Simulate slowly-varying audio context
            if args.audio_sweep:
                t = step_i / max(args.steps, 1)
                ctx[0] = ctx[1] = 0.5 + 0.5 * np.sin(2 * np.pi * t * 2)   # bass
                ctx[4] = ctx[5] = 0.5 + 0.5 * np.sin(2 * np.pi * t * 3)   # high
                ctx[10] = ctx[11] = t                                        # temperature
                ctx[12] = ctx[13] = 0.3 + 0.4 * np.sin(2 * np.pi * t * 5)  # rms
                # Occasional onset spikes
                if rng.random() < 0.05:
                    ctx[6] = ctx[7] = 0.9
                else:
                    ctx[6] = ctx[7] = max(0, ctx[6] - 0.15)
                env.set_context(ctx)

            action = rng.choice(actions, p=[0.5, 0.1, 0.1, 0.15, 0.15])
            obs, reward, done, truncated, info = env.step(action)

            # Clear screen and print
            sys.stdout.write('\033[2J\033[H')
            header = format_context_line(ctx)
            sys.stdout.write(header + '\n')
            sys.stdout.write(obs['ascii'] + '\n')
            sys.stdout.write(f'step={step_i} action={action} reward={reward:.3f}\n')
            sys.stdout.flush()
            time.sleep(args.delay)

            if done or truncated:
                break
    except KeyboardInterrupt:
        pass


def _run_live_curses(env, ctx, args):
    """Curses-based terminal mode for flicker-free rendering."""
    import curses

    def main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(args.delay * 1000))
        rng = np.random.default_rng(args.seed)
        actions = [FORWARD, FORWARD, FORWARD, TURN_LEFT, TURN_RIGHT]

        for step_i in range(args.steps):
            # Audio sweep
            if args.audio_sweep:
                t = step_i / max(args.steps, 1)
                ctx[0] = ctx[1] = 0.5 + 0.5 * np.sin(2 * np.pi * t * 2)
                ctx[4] = ctx[5] = 0.5 + 0.5 * np.sin(2 * np.pi * t * 3)
                ctx[10] = ctx[11] = t
                ctx[12] = ctx[13] = 0.3 + 0.4 * np.sin(2 * np.pi * t * 5)
                if rng.random() < 0.05:
                    ctx[6] = ctx[7] = 0.9
                else:
                    ctx[6] = ctx[7] = max(0, ctx[6] - 0.15)
                env.set_context(ctx)

            action = rng.choice(actions, p=[0.5, 0.1, 0.1, 0.15, 0.15])
            obs, reward, done, truncated, info = env.step(action)

            stdscr.erase()
            header = format_context_line(ctx)
            max_y, max_x = stdscr.getmaxyx()

            # Write header
            try:
                stdscr.addstr(0, 0, header[:max_x - 1])
            except curses.error:
                pass

            # Write frame lines
            lines = obs['ascii'].split('\n')
            for i, line in enumerate(lines):
                if i + 1 >= max_y - 1:
                    break
                try:
                    stdscr.addstr(i + 1, 0, line[:max_x - 1])
                except curses.error:
                    pass

            # Status line
            status = f'step={step_i} action={action} reward={reward:.3f}'
            try:
                stdscr.addstr(min(len(lines) + 1, max_y - 1), 0, status[:max_x - 1])
            except curses.error:
                pass

            stdscr.refresh()

            # Check for 'q' to quit
            key = stdscr.getch()
            if key == ord('q'):
                break

            if done or truncated:
                break

    curses.wrapper(main)


def _make_audio_context(
    profile: str | None,
    step_i: int,
    n_steps: int,
    rng: np.random.Generator,
    prev_ctx: np.ndarray,
) -> np.ndarray:
    """Generate a 16-float audio context for one frame.

    If *profile* is None, falls back to the original sweep-based generation.
    Otherwise applies one of the named profiles:
        high, low, ramp, pulse, random, sweep
    """
    ctx = np.zeros(16, dtype=np.float32)
    t = step_i / max(n_steps, 1)

    if profile is None:
        # Legacy default: gentle audio sweep
        ctx[0] = ctx[1] = rng.random() * 0.3 + 0.5 * np.sin(2 * np.pi * t * 2)
        ctx[2] = ctx[3] = rng.random() * 0.2 + 0.3
        ctx[4] = ctx[5] = 0.5 + 0.5 * np.sin(2 * np.pi * t * 3)
        ctx[10] = ctx[11] = t
        ctx[12] = ctx[13] = 0.3 + 0.4 * np.sin(2 * np.pi * t * 5)
        if rng.random() < 0.05:
            ctx[6] = ctx[7] = 0.9
        else:
            ctx[6] = ctx[7] = max(0, prev_ctx[6] - 0.15)

    elif profile == 'high':
        # All features 0.7-0.9, onsets 20% of frames
        for pair_start in range(0, 14, 2):
            val = rng.uniform(0.7, 0.9)
            ctx[pair_start] = ctx[pair_start + 1] = val
        # Onset: 20% fire probability
        if rng.random() < 0.20:
            ctx[6] = ctx[7] = rng.uniform(0.7, 0.95)
        else:
            ctx[6] = ctx[7] = rng.uniform(0.0, 0.05)

    elif profile == 'low':
        # All features 0.05-0.2, rare onsets
        for pair_start in range(0, 14, 2):
            val = rng.uniform(0.05, 0.2)
            ctx[pair_start] = ctx[pair_start + 1] = val
        # Onset: ~2% fire probability (rare)
        if rng.random() < 0.02:
            ctx[6] = ctx[7] = rng.uniform(0.3, 0.5)
        else:
            ctx[6] = ctx[7] = 0.0

    elif profile == 'ramp':
        # RMS ramps 0→1 over episode; other features scale proportionally
        for pair_start in range(0, 14, 2):
            base_noise = rng.uniform(-0.05, 0.05)
            ctx[pair_start] = ctx[pair_start + 1] = t + base_noise
        # RMS is the explicit ramp signal
        ctx[12] = ctx[13] = t
        # Onset scales with ramp
        if rng.random() < t * 0.15:
            ctx[6] = ctx[7] = rng.uniform(0.5, 0.9)
        else:
            ctx[6] = ctx[7] = 0.0

    elif profile == 'pulse':
        # Alternating high/low every 20 frames, onsets 20% during high
        period = 20
        is_high = (step_i // period) % 2 == 0
        if is_high:
            for pair_start in range(0, 14, 2):
                ctx[pair_start] = ctx[pair_start + 1] = rng.uniform(0.7, 0.9)
            if rng.random() < 0.20:
                ctx[6] = ctx[7] = rng.uniform(0.7, 0.95)
            else:
                ctx[6] = ctx[7] = rng.uniform(0.0, 0.05)
        else:
            for pair_start in range(0, 14, 2):
                ctx[pair_start] = ctx[pair_start + 1] = rng.uniform(0.05, 0.2)
            ctx[6] = ctx[7] = 0.0

    elif profile == 'random':
        # Uniform random [0,1] for all features, onsets 20%
        for pair_start in range(0, 14, 2):
            ctx[pair_start] = ctx[pair_start + 1] = rng.uniform(0.0, 1.0)
        if rng.random() < 0.20:
            ctx[6] = ctx[7] = rng.uniform(0.5, 1.0)
        else:
            ctx[6] = ctx[7] = rng.uniform(0.0, 0.1)

    elif profile == 'sweep':
        # Temperature sweeps 0→1, bass sweeps 1→0 (opposite directions)
        ctx[0] = ctx[1] = (1.0 - t) + rng.uniform(-0.03, 0.03)   # bass 1→0
        ctx[2] = ctx[3] = rng.uniform(0.3, 0.5)                   # mid stable
        ctx[4] = ctx[5] = 0.5 + 0.3 * np.sin(2 * np.pi * t * 4)  # high oscillates
        ctx[6] = ctx[7] = 0.0                                      # onset off by default
        ctx[8] = ctx[9] = rng.uniform(0.4, 0.6)                   # bpm stable
        ctx[10] = ctx[11] = t + rng.uniform(-0.03, 0.03)          # temperature 0→1
        ctx[12] = ctx[13] = 0.3 + 0.4 * t                         # rms rises gently
        # Sparse onset
        if rng.random() < 0.05:
            ctx[6] = ctx[7] = rng.uniform(0.5, 0.8)

    else:
        raise ValueError(f'Unknown audio profile: {profile!r}')

    return np.clip(ctx, 0.0, 1.0)


def _run_generate(args):
    """Generate frames and write as JSONL or individual text files."""
    steps = args.steps
    env = AsciiCorridorEnv(
        cols=args.cols, rows=args.rows, map_size=args.map_size,
        max_steps=steps, render_mode='ansi',
    )

    output_path = Path(args.output)
    is_jsonl = output_path.suffix in ('.jsonl', '.json', '.ndjson')

    if not is_jsonl:
        output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    actions = [FORWARD, FORWARD, FORWARD, TURN_LEFT, TURN_RIGHT]
    frame_idx = 0

    profile = getattr(args, 'audio_profile', None)

    jsonl_file = None
    if is_jsonl:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if getattr(args, 'append', False) else 'w'
        jsonl_file = open(output_path, mode, encoding='utf-8')

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep if args.seed is not None else None)
            ctx = np.zeros(16, dtype=np.float32)

            for step_i in range(steps):
                ctx = _make_audio_context(profile, step_i, steps, rng, ctx)
                env.set_context(ctx)

                action = rng.choice(actions, p=[0.5, 0.1, 0.1, 0.15, 0.15])
                obs, reward, done, truncated, info = env.step(action)

                record = {
                    'frame_idx': frame_idx,
                    'episode': ep,
                    'step': step_i,
                    'ascii_frame': obs['ascii'],
                    'audio_context': ctx.tolist(),
                    'context_line': format_context_line(ctx),
                }

                if is_jsonl:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    frame_path = output_path / f'frame_{frame_idx:06d}.txt'
                    with open(frame_path, 'w', encoding='utf-8') as f:
                        f.write(format_context_line(ctx) + '\n')
                        f.write(obs['ascii'] + '\n')

                frame_idx += 1

                if done or truncated:
                    break

            print(f'Episode {ep + 1}/{args.episodes}: {step_i + 1} steps, '
                  f'{frame_idx} total frames', file=sys.stderr)

    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    print(f'Wrote {frame_idx} frames to {output_path}', file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AURA ASCII corridor renderer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='mode')

    # ---- live mode ----
    live = sub.add_parser('live', help='Render frames live in the terminal')
    live.add_argument('--cols', type=int, default=80)
    live.add_argument('--rows', type=int, default=40)
    live.add_argument('--map-size', type=int, default=32)
    live.add_argument('--steps', type=int, default=500)
    live.add_argument('--seed', type=int, default=None)
    live.add_argument('--delay', type=float, default=0.08,
                      help='Seconds between frames')
    live.add_argument('--curses', action='store_true',
                      help='Use curses for flicker-free rendering')
    live.add_argument('--audio-sweep', action='store_true', default=True,
                      help='Sweep audio context over time (default: on)')
    live.add_argument('--no-audio-sweep', dest='audio_sweep', action='store_false')
    live.add_argument('--bass', type=float, default=None)
    live.add_argument('--rms', type=float, default=None)
    live.add_argument('--temperature', type=float, default=None)

    # ---- generate mode ----
    gen = sub.add_parser('generate', help='Generate frames as JSONL or text files')
    gen.add_argument('--cols', type=int, default=80)
    gen.add_argument('--rows', type=int, default=40)
    gen.add_argument('--map-size', type=int, default=32)
    gen.add_argument('--episodes', type=int, default=5)
    gen.add_argument('--steps', '--steps-per-episode', type=int, default=100,
                     dest='steps')
    gen.add_argument('--seed', type=int, default=42)
    gen.add_argument('--output', type=str, default='frames.jsonl',
                     help='Output path: .jsonl file or directory for .txt files')
    gen.add_argument('--append', action='store_true',
                     help='Append to output file instead of overwriting')
    gen.add_argument('--audio-profile', type=str, default=None,
                     choices=['high', 'low', 'ramp', 'pulse', 'random', 'sweep'],
                     help='Audio context profile (overrides default sweep)')

    # ---- single frame (for quick test) ----
    single = sub.add_parser('frame', help='Print a single frame and exit')
    single.add_argument('--cols', type=int, default=80)
    single.add_argument('--rows', type=int, default=40)
    single.add_argument('--map-size', type=int, default=32)
    single.add_argument('--seed', type=int, default=42)
    single.add_argument('--bass', type=float, default=0.5)
    single.add_argument('--rms', type=float, default=0.5)
    single.add_argument('--temperature', type=float, default=0.3)
    single.add_argument('--onset', type=float, default=0.0)

    args = parser.parse_args()

    if args.mode == 'live':
        _run_live(args)
    elif args.mode == 'generate':
        _run_generate(args)
    elif args.mode == 'frame':
        env = AsciiCorridorEnv(
            cols=args.cols, rows=args.rows, map_size=args.map_size,
            render_mode='ansi',
        )
        ctx = np.zeros(16, dtype=np.float32)
        ctx[0] = ctx[1] = args.bass
        ctx[12] = ctx[13] = args.rms
        ctx[10] = ctx[11] = args.temperature
        ctx[6] = ctx[7] = args.onset
        obs, _ = env.reset(seed=args.seed)
        env.set_context(ctx)
        # Step a few times to get into the corridor
        for _ in range(10):
            obs, *_ = env.step(FORWARD)
        frame = env.render()
        print(frame)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
