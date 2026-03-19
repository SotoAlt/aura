"""Episode data generation and loading for AURA training.

Generates NPZ episode files from the corridor environment + synthetic audio,
and provides a dataset loader for training batches.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


class EpisodeGenerator:
    """Generate training episodes: corridor env + synthetic audio → NPZ."""

    def __init__(self, steps_per_episode: int = 100, fps: float = 10.0,
                 audio_duration: float = 10.0, sr: int = 22050):
        self.steps_per_episode = steps_per_episode
        self.fps = fps
        self.audio_duration = audio_duration
        self.sr = sr

    def generate_episode(self, seed: int) -> dict[str, np.ndarray]:
        """Generate one episode. Memory-safe: only one episode in RAM at a time.

        Returns:
            dict with image, action, context, reward, is_first arrays.
        """
        from world_model.envs.corridor import CorridorEnv
        from world_model.audio.synthetic import SyntheticAudioGenerator
        from world_model.audio.features import AudioFeatureExtractor

        # Generate synthetic audio for this episode
        audio_gen = SyntheticAudioGenerator(sr=self.sr, episode_length=self.audio_duration)
        waveform, _ = audio_gen.generate_episode(seed=seed)

        # Extract context sequence
        extractor = AudioFeatureExtractor(sr=self.sr)
        contexts = extractor.extract_sequence(waveform, fps=self.fps)

        # Run corridor env
        env = CorridorEnv()
        obs, _ = env.reset(seed=seed)

        _zero_ctx = np.zeros(16, dtype=np.float32)

        def _get_ctx(step):
            if len(contexts) == 0:
                return _zero_ctx
            return contexts[min(step, len(contexts) - 1)]

        images = [obs['image']]
        actions = []
        rewards = []
        is_firsts = [1.0]
        context_list = [_get_ctx(0)]

        for step in range(self.steps_per_episode):
            ctx = _get_ctx(step)
            env.set_context(ctx)

            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)

            images.append(obs['image'])
            actions.append(action)
            rewards.append(reward)
            is_firsts.append(0.0)
            context_list.append(ctx)

            if term or trunc:
                break

        return {
            'image': np.array(images, dtype=np.uint8),
            'action': np.array(actions, dtype=np.int32),
            'context': np.array(context_list, dtype=np.float32),
            'reward': np.array(rewards, dtype=np.float32),
            'is_first': np.array(is_firsts, dtype=np.float32),
        }

    def generate_dataset(self, num_episodes: int, output_dir: str,
                         base_seed: int = 42):
        """Generate a dataset of episodes, one at a time (memory safe).

        Args:
            num_episodes: number of episodes to generate.
            output_dir: directory to save NPZ files.
            base_seed: starting seed.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(num_episodes), desc='Generating episodes'):
            episode = self.generate_episode(seed=base_seed + i)
            np.savez_compressed(
                output / f'episode_{i:04d}.npz',
                **episode,
            )


class NPZDataset:
    """Load NPZ episodes and serve random training batches."""

    def __init__(self, data_dir: str, seq_length: int = 10, batch_size: int = 4):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.episode_paths = sorted(self.data_dir.glob('episode_*.npz'))
        if not self.episode_paths:
            raise FileNotFoundError(f'No episodes found in {data_dir}')

    def _load_episode(self, path: Path) -> dict[str, np.ndarray]:
        data = np.load(path)
        return {k: data[k] for k in data.files}

    def _sample_sequence(self, episode: dict, rng: np.random.Generator) -> dict:
        """Sample a contiguous sequence from an episode.

        Data alignment: image[t] is the observation BEFORE action[t].
        image has T+1 frames (initial + T steps), others have T.
        We slice image[start:end] to get the pre-action observations.
        """
        T = len(episode['action'])
        max_start = max(0, T - self.seq_length)
        start = rng.integers(0, max_start + 1)
        end = start + self.seq_length

        return {
            'image': episode['image'][start:end].astype(np.float32) / 255.0,
            'action': episode['action'][start:end],
            'context': episode['context'][start:end],
            'reward': episode['reward'][start:end],
            'is_first': episode['is_first'][start:end],
        }

    def sample_batch(self, rng: np.random.Generator | None = None) -> dict:
        """Sample a random training batch.

        Returns:
            dict with arrays of shape (batch_size, seq_length, ...).
        """
        if rng is None:
            rng = np.random.default_rng()

        batch = {k: [] for k in ['image', 'action', 'context', 'reward', 'is_first']}

        for _ in range(self.batch_size):
            ep_idx = rng.integers(len(self.episode_paths))
            episode = self._load_episode(self.episode_paths[ep_idx])
            seq = self._sample_sequence(episode, rng)
            for k in batch:
                batch[k].append(seq[k])

        return {k: np.stack(v) for k, v in batch.items()}

    def __iter__(self):
        """Iterate: yield one batch per call (infinite)."""
        rng = np.random.default_rng()
        while True:
            yield self.sample_batch(rng)

    def batches(self, n: int, rng: np.random.Generator | None = None):
        """Yield exactly n batches."""
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(n):
            yield self.sample_batch(rng)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate AURA training data')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', type=str, default='../data/test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gen = EpisodeGenerator(steps_per_episode=args.steps)
    gen.generate_dataset(args.episodes, args.output, base_seed=args.seed)
    print(f'Done. {args.episodes} episodes saved to {args.output}')
