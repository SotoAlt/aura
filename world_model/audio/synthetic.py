"""Procedural audio generation for AURA training data.

Generates synthetic audio with controllable features so the world model
can learn audio→visual correlations with known ground truth.
"""

import numpy as np


class SyntheticAudioGenerator:
    """Generate procedural audio episodes with known feature profiles."""

    def __init__(self, sr: int = 22050, episode_length: float = 10.0):
        """
        Args:
            sr: Sample rate.
            episode_length: Duration of each episode in seconds.
        """
        self.sr = sr
        self.episode_length = episode_length
        self.n_samples = int(sr * episode_length)

    def _time_array(self) -> np.ndarray:
        """Precomputed time array, cached per instance."""
        if not hasattr(self, '_t_cache'):
            self._t_cache = np.arange(self.n_samples) / self.sr
        return self._t_cache

    def _bass_pulse(self, rng: np.random.Generator, freq: float = 50.0,
                    bpm: float = 120.0) -> np.ndarray:
        """Generate sub-bass pulses at a given BPM."""
        t = self._time_array()
        beat_period = 60.0 / bpm
        beat_phase = (t % beat_period) / beat_period
        envelope = np.exp(-beat_phase * 8.0)
        freq_mod = 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
        return np.sin(2 * np.pi * freq * freq_mod * t) * envelope

    def _mid_sweep(self, rng: np.random.Generator, base_freq: float = 400.0) -> np.ndarray:
        """Generate mid-frequency sweeps and pads."""
        t = self._time_array()
        # Slow frequency sweep
        sweep_rate = rng.uniform(0.1, 0.5)
        freq = base_freq * (1.0 + 0.5 * np.sin(2 * np.pi * sweep_rate * t))
        phase = np.cumsum(2 * np.pi * freq / self.sr)
        wave = np.sin(phase) * 0.3
        # Add harmonics
        wave += np.sin(phase * 2) * 0.15
        wave += np.sin(phase * 3) * 0.08
        return wave

    def _high_texture(self, rng: np.random.Generator) -> np.ndarray:
        """Generate high-frequency textures (noise bursts, shimmer)."""
        t = self._time_array()
        # Filtered noise
        noise = rng.normal(0, 0.1, self.n_samples).astype(np.float32)
        # Simple high-pass via differencing
        noise_hp = np.diff(noise, prepend=noise[0])
        noise_hp = np.diff(noise_hp, prepend=noise_hp[0])
        # Amplitude modulation for shimmer
        mod_freq = rng.uniform(2.0, 8.0)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))
        return noise_hp * envelope * 0.2

    def _beat_pattern(self, rng: np.random.Generator, bpm: float = 120.0) -> np.ndarray:
        """Generate rhythmic click/transient pattern."""
        signal = np.zeros(self.n_samples, dtype=np.float32)
        beat_samples = int(60.0 / bpm * self.sr)
        # Create click at each beat
        click_len = min(256, beat_samples // 4)
        click = np.exp(-np.arange(click_len) / (click_len / 8.0))
        click *= rng.uniform(0.3, 0.8)

        pos = 0
        while pos + click_len < self.n_samples:
            signal[pos:pos + click_len] += click
            # Sometimes add off-beat hits
            if rng.random() > 0.5:
                offbeat = pos + beat_samples // 2
                if offbeat + click_len < self.n_samples:
                    signal[offbeat:offbeat + click_len] += click * 0.5
            pos += beat_samples

        return signal

    def generate_episode(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Generate a synthetic audio episode.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (waveform, metadata) where waveform is (n_samples,) float32
            and metadata contains the generation parameters.
        """
        rng = np.random.default_rng(seed)

        # Random parameters for this episode
        bpm = rng.uniform(80, 160)
        bass_freq = rng.uniform(30, 80)
        mid_freq = rng.uniform(300, 800)
        bass_level = rng.uniform(0.3, 1.0)
        mid_level = rng.uniform(0.2, 0.8)
        high_level = rng.uniform(0.1, 0.6)
        beat_level = rng.uniform(0.2, 0.7)

        # Generate layers
        bass = self._bass_pulse(rng, freq=bass_freq, bpm=bpm) * bass_level
        mid = self._mid_sweep(rng, base_freq=mid_freq) * mid_level
        high = self._high_texture(rng) * high_level
        beats = self._beat_pattern(rng, bpm=bpm) * beat_level

        # Mix
        waveform = bass + mid + high + beats

        # Normalize to [-1, 1]
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak * 0.9

        waveform = waveform.astype(np.float32)

        metadata = {
            'bpm': bpm,
            'bass_freq': bass_freq,
            'mid_freq': mid_freq,
            'bass_level': bass_level,
            'mid_level': mid_level,
            'high_level': high_level,
            'beat_level': beat_level,
            'duration': self.episode_length,
            'sr': self.sr,
            'seed': seed,
        }

        return waveform, metadata

