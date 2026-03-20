"""DIAMOND evaluation: GIFs, audio correlation, forward motion metrics.

Runs scenarios with different audio profiles, measures:
  - Audio-visual correlation (brightness vs RMS, etc.)
  - Forward motion (mean optical flow between consecutive frames)
  - Temporal coherence (SSIM between consecutive frames)

Usage:
    python -m world_model.diamond.eval \
        --checkpoint checkpoints/diamond.ckpt \
        --data data/matsya \
        --output eval_diamond/
"""

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.diamond.sample import load_model, imagine
from world_model.eval import make_gif, measure_correlation
from world_model.eval import _make_context_sequence as _base_make_context


# ---------------------------------------------------------------------------
# Forward motion and temporal coherence metrics
# ---------------------------------------------------------------------------

def _ssim_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Compute simplified SSIM between two (H, W, 3) float images in [0, 1].

    Uses the standard formula with default constants.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / den)


def forward_motion_flow(frames: np.ndarray) -> float:
    """Mean pixel displacement between consecutive frames.

    Proxy for optical flow magnitude — higher = more forward motion.

    Args:
        frames: (T, H, W, 3) float32 in [0, 1] or uint8.

    Returns:
        Mean absolute pixel change (normalized by 255 if uint8).
    """
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0

    diffs = np.abs(np.diff(frames, axis=0))  # (T-1, H, W, 3)
    return float(np.mean(diffs))


def temporal_ssim(frames: np.ndarray) -> float:
    """Mean SSIM between consecutive frames. Higher = smoother/more coherent.

    Args:
        frames: (T, H, W, 3) float32 in [0, 1] or uint8.

    Returns:
        Mean SSIM across consecutive pairs.
    """
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0

    ssims = []
    for i in range(len(frames) - 1):
        ssims.append(_ssim_pair(frames[i], frames[i + 1]))
    return float(np.mean(ssims)) if ssims else 0.0


# ---------------------------------------------------------------------------
# Scenario context generators
# ---------------------------------------------------------------------------

def _make_context_sequence(T: int, scenario: str) -> np.ndarray:
    """Generate audio context sequence for an evaluation scenario.

    Extends the base scenarios from world_model.eval with forward_motion.
    """
    if scenario == 'forward_motion':
        ctx = np.zeros((T, 16), dtype=np.float32)
        ctx[:, 0:2] = 0.4   # bass
        ctx[:, 2:4] = 0.5   # mid
        ctx[:, 4:6] = 0.3   # high
        ctx[:, 8:10] = 0.6  # BPM
        ctx[:, 12:14] = 0.6  # RMS
        return ctx
    return _base_make_context(T, scenario)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def eval_report(checkpoint_path: str, data_dir: str, output_dir: str,
                n_frames: int = 50):
    """Run full DIAMOND evaluation.

    Args:
        checkpoint_path: Path to trained checkpoint.
        data_dir: Path to NPZ episodes (for seed frames).
        output_dir: Output directory for GIFs and metrics.
        n_frames: Frames per scenario.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f'Evaluating DIAMOND checkpoint: {checkpoint_path}')
    print(f'Data: {data_dir}')
    print(f'Output: {output_dir}')
    print(f'Frames per scenario: {n_frames}')
    print()

    # Load model
    model, cfg = load_model(checkpoint_path)
    device = next(model.parameters()).device
    C = cfg['context_frames']

    # Load seed frames from first episode
    episodes = sorted(Path(data_dir).glob('episode_*.npz'))
    if not episodes:
        raise FileNotFoundError(f'No episodes in {data_dir}')
    ep = np.load(episodes[len(episodes) // 2])
    seed_frames = ep['image'][:C]  # (C, H, W, 3) uint8

    scenarios = ['high_energy', 'low_energy', 'beat_onset', 'rms_ramp',
                 'forward_motion']
    all_metrics = {}

    for scenario in scenarios:
        print(f'Scenario: {scenario}...', end=' ', flush=True)

        ctx = _make_context_sequence(n_frames, scenario)
        frames = imagine(model, seed_frames, ctx, n_frames, device)
        # frames: (n_frames, H, W, 3) uint8

        # Save GIF
        gif_path = output / f'{scenario}.gif'
        make_gif(frames, str(gif_path), fps=10, scale=4)

        # Standard correlation metrics
        frames_f = frames.astype(np.float32) / 255.0
        metrics = measure_correlation(frames_f, ctx)

        # DIAMOND-specific metrics
        metrics['forward_motion_flow'] = forward_motion_flow(frames)
        metrics['temporal_ssim'] = temporal_ssim(frames)

        all_metrics[scenario] = metrics

        print(f'brightness={metrics["mean_brightness"]:.3f}, '
              f'flow={metrics["forward_motion_flow"]:.4f}, '
              f'ssim={metrics["temporal_ssim"]:.3f}')

    # Cross-scenario comparisons
    he = all_metrics.get('high_energy', {})
    le = all_metrics.get('low_energy', {})
    fm = all_metrics.get('forward_motion', {})

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

    # DIAMOND success criteria (v0.3 thesis test)
    criteria = {
        'no_nan_or_inf': (
            not summary['cross_scenario']['any_nan']
            and not summary['cross_scenario']['any_inf']
        ),
        'brightness_rms_corr_gt_0.3': (
            all_metrics.get('rms_ramp', {}).get('brightness_rms_corr', 0) > 0.3
        ),
        'high_brighter_than_low': (
            summary['cross_scenario']['high_vs_low_brightness_diff'] > 0
        ),
        'forward_motion_flow_gt_0.01': (
            fm.get('forward_motion_flow', 0) > 0.01
        ),
        'temporal_ssim_gt_0.7': (
            fm.get('temporal_ssim', 0) > 0.7
        ),
    }
    summary['diamond_criteria'] = criteria

    # Save
    metrics_path = output / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print('=' * 55)
    print('DIAMOND v0.3 Success Criteria:')
    for k, v in criteria.items():
        status = 'PASS' if v else 'FAIL'
        print(f'  [{status}] {k}')
    all_pass = all(criteria.values())
    print(f'\n  {"THESIS VALIDATED" if all_pass else "NEEDS WORK"}')
    print('=' * 55)
    print(f'\nMetrics saved to {metrics_path}')
    print(f'GIFs saved to {output_dir}/')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DIAMOND world model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True,
                        help='Path to NPZ episodes (for seed frames)')
    parser.add_argument('--output', type=str, default='eval_diamond/')
    parser.add_argument('--frames', type=int, default=50)
    args = parser.parse_args()

    eval_report(args.checkpoint, args.data, args.output, args.frames)
