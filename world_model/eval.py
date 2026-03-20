"""AURA evaluation script.

Loads a checkpoint, runs imagination with various audio contexts,
generates GIFs, and measures audio-visual correlation.

Usage:
    python -m world_model.eval --checkpoint checkpoints/latest.ckpt --output eval_output/
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from world_model.dreamer.agent import (
    load_config, make_rssm_config, imagine_trajectory,
)
from world_model.dreamer.rssm import initial_state
from world_model.dreamer.checkpoint import load_checkpoint


def imagine_from_params(params: dict, cfg: dict, contexts: np.ndarray,
                        actions: np.ndarray | None = None) -> np.ndarray:
    """Imagine a trajectory from loaded params.

    Args:
        params: World model parameters.
        cfg: Training config dict.
        contexts: (T, 16) or (B, T, 16) audio context array.
        actions: (T,) or (B, T) int actions. If None, uses action 0 (forward).

    Returns:
        (B, T, H, W, 3) predicted frames in [0, 1].
    """
    rssm_cfg = make_rssm_config(cfg)

    # Ensure batch dim
    if contexts.ndim == 2:
        contexts = contexts[None]  # (1, T, 16)
    B, T = contexts.shape[:2]

    # Build actions
    if actions is None:
        actions = np.zeros((B, T), dtype=np.int32)
    if actions.ndim == 1:
        actions = actions[None]
    actions_oh = jax.nn.one_hot(jnp.array(actions), cfg['action_dim'])

    # Imagine
    init_s = initial_state(rssm_cfg, B)
    rng = jax.random.key(0)
    frames = imagine_trajectory(
        params, rssm_cfg, init_s,
        actions_oh, jnp.array(contexts), rng,
    )
    return np.array(frames)


def make_gif(frames: np.ndarray, path: str, fps: int = 10, scale: int = 4):
    """Save frames as animated GIF.

    Args:
        frames: (T, H, W, 3) float32 [0,1] or uint8.
        path: Output file path.
        fps: Frames per second.
        scale: Upscale factor for visibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)

    images = []
    H, W = frames.shape[1], frames.shape[2]
    for f in frames:
        img = Image.fromarray(f)
        if scale > 1:
            img = img.resize((W * scale, H * scale), Image.NEAREST)
        images.append(img)

    duration = int(1000 / fps)
    images[0].save(
        str(path), save_all=True, append_images=images[1:],
        duration=duration, loop=0,
    )


def measure_correlation(frames: np.ndarray, contexts: np.ndarray) -> dict:
    """Measure audio-visual correlation metrics.

    Args:
        frames: (T, 64, 64, 3) float32 in [0, 1].
        contexts: (T, 16) float32.

    Returns:
        Dict of correlation metrics.
    """
    T = min(len(frames), len(contexts))
    frames = frames[:T]
    contexts = contexts[:T]

    # Frame brightness (mean pixel value)
    brightness = np.mean(frames, axis=(1, 2, 3))  # (T,)

    # RMS energy from context (indices 12-13)
    rms = (contexts[:, 12] + contexts[:, 13]) / 2  # (T,)

    # Frame deltas (temporal change)
    frame_deltas = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2, 3))  # (T-1,)

    # Onset from context (indices 6-7)
    onsets = (contexts[:, 6] + contexts[:, 7]) / 2  # (T,)

    # Spectral centroid / temperature (indices 10-11)
    temperature = (contexts[:, 10] + contexts[:, 11]) / 2  # (T,)

    # Pearson correlations
    def _corr(a, b):
        n = min(len(a), len(b))
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(r) else float(r)

    # Red channel mean as proxy for "temperature"
    red_channel = np.mean(frames[:, :, :, 0], axis=(1, 2))

    metrics = {
        'brightness_rms_corr': _corr(brightness, rms),
        'delta_onset_corr': _corr(frame_deltas, onsets[1:]),
        'red_temperature_corr': _corr(red_channel, temperature),
        'mean_brightness': float(np.mean(brightness)),
        'brightness_std': float(np.std(brightness)),
        'mean_frame_delta': float(np.mean(frame_deltas)),
        'has_nan': bool(np.any(np.isnan(frames))),
        'has_inf': bool(np.any(np.isinf(frames))),
        'min_pixel': float(np.min(frames)),
        'max_pixel': float(np.max(frames)),
    }
    return metrics


def _make_context_sequence(T: int, scenario: str) -> np.ndarray:
    """Generate a context sequence for a specific evaluation scenario."""
    ctx = np.zeros((T, 16), dtype=np.float32)

    if scenario == 'high_energy':
        # All features high
        ctx[:, 0:14:2] = 0.9  # raw
        ctx[:, 1:14:2] = 0.9  # ema

    elif scenario == 'low_energy':
        # All features low / silence
        ctx[:, 0:14:2] = 0.05
        ctx[:, 1:14:2] = 0.05

    elif scenario == 'beat_onset':
        # Periodic onsets every 5 frames
        for t in range(0, T, 5):
            ctx[t, 6] = 1.0  # onset raw
            ctx[t, 7] = 0.8  # onset ema
        ctx[:, 12:14] = 0.5  # moderate RMS

    elif scenario == 'sweep':
        # Gradual increase in temperature (spectral centroid)
        ramp = np.linspace(0, 1, T)
        ctx[:, 10] = ramp  # centroid raw
        ctx[:, 11] = ramp  # centroid ema
        ctx[:, 12:14] = 0.5  # moderate RMS

    elif scenario == 'rms_ramp':
        # RMS energy ramp from 0 to 1
        ramp = np.linspace(0, 1, T)
        ctx[:, 12] = ramp
        ctx[:, 13] = ramp

    return ctx


def eval_report(checkpoint_path: str, output_dir: str, n_frames: int = 50):
    """Run full evaluation: all scenarios, GIFs, correlation metrics.

    Args:
        checkpoint_path: Path to checkpoint.
        output_dir: Directory for output GIFs and metrics.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    scenarios = ['high_energy', 'low_energy', 'beat_onset', 'sweep', 'rms_ramp']
    all_metrics = {}

    print(f'Evaluating checkpoint: {checkpoint_path}')
    print(f'Output: {output_dir}')
    print(f'Frames per scenario: {n_frames}')
    print()

    # Load checkpoint once
    ckpt = load_checkpoint(checkpoint_path)
    params, cfg = ckpt['params'], ckpt['cfg']

    for scenario in scenarios:
        print(f'Scenario: {scenario}...', end=' ')

        # Generate context
        ctx = _make_context_sequence(n_frames, scenario)

        # Imagine trajectory
        frames = imagine_from_params(params, cfg, ctx)
        frames_single = frames[0]  # (T, 64, 64, 3)

        # Save GIF
        gif_path = output / f'{scenario}.gif'
        make_gif(frames_single, str(gif_path), fps=10)

        # Measure correlation
        metrics = measure_correlation(frames_single, ctx)
        all_metrics[scenario] = metrics

        print(f'brightness={metrics["mean_brightness"]:.3f}, '
              f'rms_corr={metrics["brightness_rms_corr"]:.3f}, '
              f'nan={metrics["has_nan"]}')

    # Cross-scenario comparisons
    he = all_metrics.get('high_energy', {})
    le = all_metrics.get('low_energy', {})

    summary = {
        'scenarios': all_metrics,
        'cross_scenario': {
            'high_vs_low_brightness_diff': (
                he.get('mean_brightness', 0) - le.get('mean_brightness', 0)
            ),
            'any_nan': any(m.get('has_nan', False) for m in all_metrics.values()),
            'any_inf': any(m.get('has_inf', False) for m in all_metrics.values()),
        },
    }

    # P1 success criteria checks
    criteria = {
        'no_nan_or_inf': not summary['cross_scenario']['any_nan']
                         and not summary['cross_scenario']['any_inf'],
        'brightness_rms_corr_gt_0.3': (
            all_metrics.get('rms_ramp', {}).get('brightness_rms_corr', 0) > 0.3
        ),
        'high_brighter_than_low': (
            summary['cross_scenario']['high_vs_low_brightness_diff'] > 0
        ),
    }
    summary['p1_criteria'] = criteria

    # Save
    metrics_path = output / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print('=' * 50)
    print('P1 Success Criteria:')
    for k, v in criteria.items():
        status = 'PASS' if v else 'FAIL'
        print(f'  [{status}] {k}')
    print('=' * 50)
    print(f'\nMetrics saved to {metrics_path}')
    print(f'GIFs saved to {output_dir}/')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate AURA world model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='eval_output/',
                        help='Output directory for GIFs and metrics')
    parser.add_argument('--frames', type=int, default=50,
                        help='Frames per scenario')
    args = parser.parse_args()

    eval_report(args.checkpoint, args.output, n_frames=args.frames)
