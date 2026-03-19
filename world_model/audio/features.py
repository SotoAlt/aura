"""Audio feature extraction for AURA world model conditioning.

Extracts a 16-float context vector from audio:
  [0-1]   Sub-bass energy (20-80 Hz)      — raw + EMA
  [2-3]   Mid energy (250 Hz - 2 kHz)     — raw + EMA
  [4-5]   High frequency (4-20 kHz)       — raw + EMA
  [6-7]   Onset detection                 — raw + EMA
  [8-9]   Estimated BPM (normalized)      — raw + EMA
  [10-11]  Spectral centroid               — raw + EMA
  [12-13]  RMS energy                      — raw + EMA
  [14-15]  Reserved                        — zeros
"""

import numpy as np
import librosa


def unpack_context(c: np.ndarray) -> dict[str, float]:
    """Unpack 16-float context vector into named audio features.

    Each feature has raw (even index) and EMA-smoothed (odd index) values;
    this returns their average.
    """
    return {
        'bass': float((c[0] + c[1]) / 2),
        'mid': float((c[2] + c[3]) / 2),
        'high': float((c[4] + c[5]) / 2),
        'onset': float((c[6] + c[7]) / 2),
        'bpm': float((c[8] + c[9]) / 2),
        'temperature': float((c[10] + c[11]) / 2),
        'rms': float((c[12] + c[13]) / 2),
    }


class AudioFeatureExtractor:
    """Extract 16-float audio context vectors from audio signals."""

    def __init__(self, sr: int = 22050, hop_length: int = 512, ema_alpha: float = 0.3):
        self.sr = sr
        self.hop_length = hop_length
        self.ema_alpha = ema_alpha
        self._ema_state = np.zeros(7, dtype=np.float32)

    def reset(self):
        """Reset EMA state between episodes."""
        self._ema_state = np.zeros(7, dtype=np.float32)

    def _band_energy(self, S: np.ndarray, freqs: np.ndarray,
                     low: float, high: float) -> float:
        """Mean energy in a frequency band from magnitude spectrogram."""
        mask = (freqs >= low) & (freqs <= high)
        if not mask.any():
            return 0.0
        return float(np.mean(S[mask]))

    def _normalize(self, value: float, vmin: float, vmax: float) -> float:
        """Clip and normalize to [0, 1]."""
        if vmax <= vmin:
            return 0.0
        return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))

    def extract_frame(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Extract a single 16-float context vector from an audio chunk.

        Args:
            audio_chunk: 1D audio signal (mono), any length.

        Returns:
            (16,) float32 array with values in [0, 1].
        """
        if len(audio_chunk) < 512:
            audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))

        # Magnitude spectrogram (single frame or short segment)
        S = np.abs(librosa.stft(audio_chunk, n_fft=1024, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=1024)
        S_mean = S.mean(axis=1)  # average across time frames

        # 1. Sub-bass (20-80 Hz)
        sub_bass = self._normalize(self._band_energy(S_mean, freqs, 20, 80), 0, 2.0)

        # 2. Mid (250 Hz - 2 kHz)
        mid = self._normalize(self._band_energy(S_mean, freqs, 250, 2000), 0, 1.0)

        # 3. High (4-20 kHz)
        high = self._normalize(self._band_energy(S_mean, freqs, 4000, 20000), 0, 0.5)

        # 4. Onset strength (mean over chunk)
        onset_env = librosa.onset.onset_strength(
            y=audio_chunk, sr=self.sr, hop_length=self.hop_length
        )
        onset = self._normalize(float(np.mean(onset_env)), 0, 5.0)

        # 5. BPM estimate (normalized: 60-200 BPM → [0, 1])
        if len(audio_chunk) > self.sr:
            tempo = librosa.beat.tempo(y=audio_chunk, sr=self.sr, hop_length=self.hop_length)
            bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        else:
            bpm = 120.0  # default for short chunks
        bpm_norm = self._normalize(bpm, 60.0, 200.0)

        # 6. Spectral centroid (normalized by Nyquist)
        centroid = librosa.feature.spectral_centroid(
            y=audio_chunk, sr=self.sr, hop_length=self.hop_length
        )
        centroid_norm = self._normalize(
            float(np.mean(centroid)), 0, self.sr / 2
        )

        # 7. RMS energy
        rms = librosa.feature.rms(y=audio_chunk, hop_length=self.hop_length)
        rms_norm = self._normalize(float(np.mean(rms)), 0, 0.5)

        raw = np.array([sub_bass, mid, high, onset, bpm_norm, centroid_norm, rms_norm],
                       dtype=np.float32)

        # EMA smoothing
        self._ema_state = self.ema_alpha * raw + (1 - self.ema_alpha) * self._ema_state

        # Interleave raw and EMA: [raw0, ema0, raw1, ema1, ...]
        context = np.zeros(16, dtype=np.float32)
        context[0:14:2] = raw
        context[1:14:2] = self._ema_state

        return context

    def extract_sequence(self, audio: np.ndarray, fps: float = 10.0) -> np.ndarray:
        """Extract context vectors for an entire audio signal at a given FPS.

        Args:
            audio: 1D mono audio signal.
            fps: Frames per second for context extraction.

        Returns:
            (T, 16) float32 array.
        """
        self.reset()
        chunk_size = int(self.sr / fps)
        n_frames = max(1, len(audio) // chunk_size)

        contexts = np.zeros((n_frames, 16), dtype=np.float32)
        for i in range(n_frames):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio[start:end]
            contexts[i] = self.extract_frame(chunk)

        return contexts
