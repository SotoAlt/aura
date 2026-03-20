"""Video data pipeline for AURA training.

Downloads a YouTube video, extracts frames + audio, applies audio-driven
augmentations, and builds NPZ episodes compatible with NPZDataset.

The video's visuals are NOT audio-reactive, so we apply augmentations that
transform each frame based on its paired audio context — this teaches the
model that audio drives visual changes.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: Download & extract
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: str, fps: float = 10.0,
                   size: int = 64) -> tuple[Path, Path]:
    """Download video, extract frames at fps and audio as mono WAV.

    Returns:
        (frames_dir, audio_path)
    """
    out = Path(output_dir)
    raw_dir = out / '_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    video_path = raw_dir / 'video.mp4'
    audio_path = raw_dir / 'audio.wav'
    frames_dir = raw_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)

    # Download with yt-dlp
    if not video_path.exists():
        print(f'Downloading video from {url} ...')
        subprocess.run([
            sys.executable, '-m', 'yt_dlp',
            '-f', 'best[ext=mp4]/best',
            '-o', str(video_path),
            url,
        ], check=True)
    else:
        print(f'Video already downloaded: {video_path}')

    # Extract audio (mono, 22050 Hz)
    if not audio_path.exists():
        print('Extracting audio ...')
        subprocess.run([
            'ffmpeg', '-y', '-i', str(video_path),
            '-ar', '22050', '-ac', '1', str(audio_path),
        ], check=True)

    # Extract frames — center-crop to square, resize to size×size
    existing = sorted(frames_dir.glob('frame_*.png'))
    if not existing:
        print(f'Extracting frames at {fps} fps ...')
        crop_scale = (
            f'crop=min(iw\\,ih):min(iw\\,ih),'
            f'scale={size}:{size}:flags=lanczos'
        )
        subprocess.run([
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', crop_scale,
            '-r', str(fps),
            str(frames_dir / 'frame_%05d.png'),
        ], check=True)
    else:
        print(f'{len(existing)} frames already extracted')

    return frames_dir, audio_path


def load_frames(frames_dir: Path) -> np.ndarray:
    """Load all extracted PNG frames as (N, H, W, 3) uint8 array."""
    paths = sorted(frames_dir.glob('frame_*.png'))
    if not paths:
        raise FileNotFoundError(f'No frames in {frames_dir}')
    frames = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        frames.append(np.array(img, dtype=np.uint8))
    return np.stack(frames)


# ---------------------------------------------------------------------------
# Step 2: Audio-driven augmentation
# ---------------------------------------------------------------------------

def augment_frame(frame: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Transform a frame based on audio context to create audio-visual correlation.

    Args:
        frame: (H, W, 3) uint8
        context: (16,) float32 audio context vector

    Returns:
        (H, W, 3) uint8 augmented frame
    """
    from world_model.audio.features import unpack_context

    feat = unpack_context(context)
    img = frame.astype(np.float32)

    # 1. Brightness from RMS energy: dim at low energy, bright at high
    brightness = 0.4 + feat['rms'] * 0.8  # range [0.4, 1.2]
    img = img * brightness

    # 2. Color temperature from spectral centroid
    #    High centroid → warm (boost red, reduce blue)
    #    Low centroid → cool (boost blue, reduce red)
    temp_shift = (feat['temperature'] - 0.5) * 0.4  # [-0.2, 0.2]
    img[:, :, 0] = img[:, :, 0] * (1.0 + temp_shift)  # red
    img[:, :, 2] = img[:, :, 2] * (1.0 - temp_shift)  # blue

    # 3. Saturation boost from bass energy
    gray = np.mean(img, axis=2, keepdims=True)
    sat_factor = 1.0 + feat['bass'] * 0.6  # [1.0, 1.6]
    img = gray + (img - gray) * sat_factor

    # 4. Contrast from onset detection
    mean_val = np.mean(img)
    contrast = 1.0 + feat['onset'] * 0.5  # [1.0, 1.5]
    img = mean_val + (img - mean_val) * contrast

    # 5. Subtle grain from high frequency energy
    if feat['high'] > 0.1:
        noise_scale = feat['high'] * 15.0  # up to 15 intensity units
        noise = np.random.standard_normal(img.shape).astype(np.float32) * noise_scale
        img = img + noise

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 2.5: Optical flow actions
# ---------------------------------------------------------------------------

def compute_flow_actions(frames: np.ndarray) -> np.ndarray:
    """Compute motion actions from consecutive frame differences.

    Returns (N-1, 3) float32 array:
        [0] forward_speed — overall motion magnitude (0-1)
        [1] horizontal_shift — left/right bias (-1 to 1)
        [2] vertical_shift — up/down bias (-1 to 1)

    Uses simple frame differencing (fast, no OpenCV needed).
    """
    n = len(frames) - 1
    actions = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        # Frame difference as float
        diff = frames[i + 1].astype(np.float32) - frames[i].astype(np.float32)
        # diff shape: (H, W, 3)

        # Forward speed: mean absolute pixel change, normalized
        speed = np.mean(np.abs(diff)) / 255.0
        actions[i, 0] = np.clip(speed * 10.0, 0, 1)  # scale up, most changes are small

        # Horizontal shift: compare left half vs right half motion
        H, W = diff.shape[:2]
        left_energy = np.mean(np.abs(diff[:, :W // 2]))
        right_energy = np.mean(np.abs(diff[:, W // 2:]))
        total = left_energy + right_energy + 1e-8
        actions[i, 1] = (right_energy - left_energy) / total  # -1 to 1

        # Vertical shift: compare top half vs bottom half motion
        top_energy = np.mean(np.abs(diff[:H // 2, :]))
        bottom_energy = np.mean(np.abs(diff[H // 2:, :]))
        total = top_energy + bottom_energy + 1e-8
        actions[i, 2] = (bottom_energy - top_energy) / total  # -1 to 1

    return actions


# ---------------------------------------------------------------------------
# Step 3: Synthetic context profiles for augmentation passes
# ---------------------------------------------------------------------------

def _make_synthetic_contexts(n_frames: int, profile: str,
                             seed: int) -> np.ndarray:
    """Generate a synthetic (16, ) context sequence for augmentation variety.

    Profiles:
        high   — sustained high energy
        low    — sustained low energy
        ramp   — energy ramps from low to high
        pulse  — alternating high/low every ~2 seconds (at 10 fps)
    """
    rng = np.random.default_rng(seed)
    contexts = np.zeros((n_frames, 16), dtype=np.float32)

    if profile == 'high':
        base = rng.uniform(0.6, 0.9, size=7).astype(np.float32)
    elif profile == 'low':
        base = rng.uniform(0.05, 0.25, size=7).astype(np.float32)
    elif profile == 'ramp':
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            vals = rng.uniform(0.05, 0.15, size=7).astype(np.float32) + t * 0.7
            contexts[i, 0:14:2] = vals
            contexts[i, 1:14:2] = vals
        return contexts
    elif profile == 'pulse':
        period = 20  # ~2 sec at 10 fps
        for i in range(n_frames):
            on = (i // period) % 2 == 0
            vals = rng.uniform(0.6, 0.9, size=7) if on else rng.uniform(0.05, 0.2, size=7)
            vals = vals.astype(np.float32)
            contexts[i, 0:14:2] = vals
            contexts[i, 1:14:2] = vals
        return contexts
    else:
        base = rng.uniform(0.2, 0.6, size=7).astype(np.float32)

    # For constant profiles (high/low), repeat with slight per-frame jitter
    for i in range(n_frames):
        jitter = rng.uniform(-0.05, 0.05, size=7).astype(np.float32)
        vals = np.clip(base + jitter, 0, 1)
        contexts[i, 0:14:2] = vals
        contexts[i, 1:14:2] = vals

    return contexts


# ---------------------------------------------------------------------------
# Step 4: Build episodes
# ---------------------------------------------------------------------------

def build_episodes(frames_dir: Path, audio_path: Path, output_dir: str,
                   steps_per_episode: int = 100, augmentation_passes: int = 5,
                   fps: float = 10.0):
    """Build NPZ episodes from extracted video frames + audio.

    Creates sliding-window episodes from the frame sequence, each with
    `augmentation_passes` variants using different audio contexts.

    Args:
        frames_dir: Directory of extracted PNG frames.
        audio_path: Path to extracted WAV audio.
        output_dir: Where to write episode_XXXX.npz files.
        steps_per_episode: Frames per episode (actions = steps, images = steps+1).
        augmentation_passes: Number of augmentation variants per window.
        fps: Frame extraction rate (must match download_video fps).
    """
    import librosa
    from world_model.audio.features import AudioFeatureExtractor

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load frames
    print('Loading frames ...')
    all_frames = load_frames(frames_dir)
    n_total = len(all_frames)
    print(f'  {n_total} frames loaded ({all_frames.shape})')

    # Extract real audio contexts
    print('Extracting audio features ...')
    audio, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    extractor = AudioFeatureExtractor(sr=22050)
    real_contexts = extractor.extract_sequence(audio, fps=fps)
    print(f'  {len(real_contexts)} context vectors extracted')

    # Align: use min of frames and contexts
    n_usable = min(n_total, len(real_contexts))
    all_frames = all_frames[:n_usable]
    real_contexts = real_contexts[:n_usable]

    # Compute optical flow actions from raw frames
    print('Computing flow actions ...')
    all_flow_actions = compute_flow_actions(all_frames)
    print(f'  flow: mean_speed={all_flow_actions[:, 0].mean():.3f}, '
          f'range=[{all_flow_actions[:, 0].min():.3f}, {all_flow_actions[:, 0].max():.3f}]')

    # Sliding window
    window = steps_per_episode + 1  # images has steps+1 frames
    stride = 20
    starts = list(range(0, n_usable - window + 1, stride))
    if not starts:
        # Video too short for stride — use single episode
        starts = [0]
        window = min(window, n_usable)
        steps_per_episode = window - 1
    print(f'  {len(starts)} base windows (stride={stride})')

    profiles = ['real', 'high', 'low', 'ramp', 'pulse']
    ep_idx = 0

    for win_i, start in enumerate(tqdm(starts, desc='Building episodes')):
        end = start + window
        window_frames = all_frames[start:end]
        window_contexts = real_contexts[start:end]

        for pass_i in range(min(augmentation_passes, len(profiles))):
            profile = profiles[pass_i]

            if profile == 'real':
                contexts = window_contexts
            else:
                contexts = _make_synthetic_contexts(
                    len(window_frames), profile, seed=win_i * 100 + pass_i
                )

            # Augment each frame
            aug_frames = np.empty_like(window_frames)
            for fi in range(len(window_frames)):
                aug_frames[fi] = augment_frame(window_frames[fi], contexts[fi])

            # Flow actions for this window (continuous 3-float vectors)
            window_actions = all_flow_actions[start:start + steps_per_episode]

            # Build NPZ episode
            episode = {
                'image': aug_frames,                                       # (T+1, H, W, 3) uint8
                'action': window_actions.astype(np.float32),               # (T, 3) — flow actions
                'context': contexts[:window].astype(np.float32),           # (T+1, 16)
                'reward': np.zeros(steps_per_episode, dtype=np.float32),   # (T,)
                'is_first': np.concatenate([
                    np.array([1.0], dtype=np.float32),
                    np.zeros(steps_per_episode, dtype=np.float32),
                ]),                                                        # (T+1,)
            }

            np.savez_compressed(out / f'episode_{ep_idx:04d}.npz', **episode)
            ep_idx += 1

    print(f'Done. {ep_idx} episodes saved to {out}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build AURA training episodes from a YouTube video'
    )
    parser.add_argument('--url', type=str, required=True,
                        help='YouTube video URL')
    parser.add_argument('--output', type=str, default='data/matsya',
                        help='Output directory for episodes')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Frame extraction rate')
    parser.add_argument('--steps', type=int, default=100,
                        help='Steps per episode')
    parser.add_argument('--passes', type=int, default=5,
                        help='Augmentation passes per window')
    parser.add_argument('--size', type=int, default=64,
                        help='Frame size (square)')
    args = parser.parse_args()

    frames_dir, audio_path = download_video(
        args.url, args.output, fps=args.fps, size=args.size
    )
    build_episodes(
        frames_dir, audio_path, args.output,
        steps_per_episode=args.steps,
        augmentation_passes=args.passes,
        fps=args.fps,
    )
