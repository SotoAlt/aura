"""AURA P0 end-to-end verification.

Run: python -m world_model.test_pipeline
From repo root with venv activated.
"""

import sys
import time
import shutil
import tempfile
import numpy as np

start = time.time()


def test_audio_features():
    """Verify 16-float output in [0, 1]."""
    print('Testing audio features...', end=' ')
    from world_model.audio.features import AudioFeatureExtractor

    extractor = AudioFeatureExtractor(sr=22050)

    # Single frame
    chunk = np.random.randn(2205).astype(np.float32)  # 100ms at 22050 Hz
    ctx = extractor.extract_frame(chunk)
    assert ctx.shape == (16,), f'Expected (16,), got {ctx.shape}'
    assert ctx.dtype == np.float32
    assert np.all(ctx >= 0.0) and np.all(ctx <= 1.0), f'Values out of range: {ctx}'

    # Sequence
    audio = np.random.randn(22050 * 3).astype(np.float32)  # 3 seconds
    seq = extractor.extract_sequence(audio, fps=10.0)
    assert seq.shape[0] > 0, 'Empty sequence'
    assert seq.shape[1] == 16
    assert np.all(seq >= 0.0) and np.all(seq <= 1.0)
    print(f'OK — frame {ctx.shape}, sequence {seq.shape}')


def test_synthetic_audio():
    """Verify synthetic audio generation."""
    print('Testing synthetic audio...', end=' ')
    from world_model.audio.synthetic import SyntheticAudioGenerator

    gen = SyntheticAudioGenerator(sr=22050, episode_length=5.0)
    waveform, meta = gen.generate_episode(seed=42)
    assert waveform.shape == (22050 * 5,), f'Expected {22050*5}, got {waveform.shape}'
    assert waveform.dtype == np.float32
    assert np.max(np.abs(waveform)) <= 1.0
    assert 'bpm' in meta
    print(f'OK — {waveform.shape[0]} samples, BPM={meta["bpm"]:.0f}')


def test_corridor_env():
    """Verify Gymnasium API and obs shapes."""
    print('Testing corridor env...', end=' ')
    from world_model.envs.corridor import CorridorEnv

    env = CorridorEnv()
    obs, info = env.reset(seed=42)
    assert obs['image'].shape == (64, 64, 3), f'Image shape: {obs["image"].shape}'
    assert obs['image'].dtype == np.uint8
    assert obs['context'].shape == (16,)

    # Run 100 steps
    for _ in range(100):
        ctx = np.random.rand(16).astype(np.float32)
        env.set_context(ctx)
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        assert obs['image'].shape == (64, 64, 3)

    print(f'OK — 100 steps, image {obs["image"].shape}')


def test_data_generation():
    """Generate 5 test episodes and verify format."""
    print('Testing data generation...', end=' ')
    from world_model.data.generate import EpisodeGenerator, NPZDataset

    tmpdir = tempfile.mkdtemp()
    try:
        gen = EpisodeGenerator(steps_per_episode=20, audio_duration=2.0)
        gen.generate_dataset(5, tmpdir, base_seed=42)

        # Verify files exist
        npz_files = list(sorted(p for p in __import__('pathlib').Path(tmpdir).glob('*.npz')))
        assert len(npz_files) == 5, f'Expected 5 episodes, got {len(npz_files)}'

        # Load and verify format
        data = np.load(npz_files[0])
        assert 'image' in data and 'action' in data and 'context' in data
        assert data['image'].dtype == np.uint8
        assert data['context'].shape[1] == 16

        # Test dataset loader
        ds = NPZDataset(tmpdir, seq_length=10, batch_size=2)
        batch = ds.sample_batch()
        assert batch['image'].shape == (2, 10, 64, 64, 3)
        assert batch['context'].shape == (2, 10, 16)
        print(f'OK — 5 episodes, batch image {batch["image"].shape}')
        return tmpdir  # keep for RSSM test
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def test_rssm_shapes():
    """Verify cRSSM forward pass dimensions."""
    print('Testing cRSSM shapes...', end=' ')
    import jax
    import jax.numpy as jnp
    from world_model.dreamer.rssm import (
        RSSMConfig, init_rssm, initial_state, obs_step, img_step, get_features,
    )

    cfg = RSSMConfig(
        deter_dim=256, stoch_dim=16, classes=16, hidden_dim=128,
        embed_dim=256, action_dim=3, context_dim=16,
    )
    rng = jax.random.key(0)
    k1, k2, k3 = jax.random.split(rng, 3)

    params = init_rssm(k1, cfg)
    state = initial_state(cfg, batch_size=2)

    action = jax.nn.one_hot(jnp.array([0, 1]), 3)
    context = jnp.ones((2, 16)) * 0.5
    embed = jnp.ones((2, 256)) * 0.1
    is_first = jnp.array([1.0, 0.0])

    # Prior step
    new_state = img_step(params, cfg, state, action, context, k2)
    assert new_state.deter.shape == (2, 256)
    assert new_state.stoch.shape == (2, 16, 16)

    # Posterior step
    post, prior = obs_step(params, cfg, state, action, embed, is_first, context, k3)
    assert post.deter.shape == (2, 256)

    # Features
    feat = get_features(post)
    assert feat.shape == (2, 256 + 16 * 16)
    print(f'OK — deter {new_state.deter.shape}, features {feat.shape}')


def test_full_pipeline(data_dir: str):
    """Init world model, load batch, run 1 training step on CPU."""
    print('Testing full training pipeline...', end=' ')
    import jax
    import jax.numpy as jnp
    from world_model.dreamer.agent import Trainer, load_config
    from world_model.data.generate import NPZDataset

    cfg = load_config('aura_debug')
    trainer = Trainer(cfg)

    rng = jax.random.key(42)
    params, opt_state = trainer.init(rng)

    # Load a batch
    ds = NPZDataset(data_dir, seq_length=cfg['seq_length'], batch_size=cfg['batch_size'])
    batch_np = ds.sample_batch()

    # Convert to JAX
    batch = {k: jnp.array(v) for k, v in batch_np.items()}

    # Run 1 training step
    rng, step_rng = jax.random.split(rng)
    new_params, new_opt_state, metrics = trainer.train_step(
        params, opt_state, batch, step_rng
    )

    # Verify
    assert jnp.isfinite(metrics['total_loss']), f'Loss not finite: {metrics["total_loss"]}'
    assert jnp.isfinite(metrics['grad_norm']), f'Grad norm not finite: {metrics["grad_norm"]}'
    assert metrics['total_loss'] > 0, 'Loss should be positive'

    print(f'OK — loss={float(metrics["total_loss"]):.4f}, '
          f'grad_norm={float(metrics["grad_norm"]):.4f}')
    return metrics


if __name__ == '__main__':
    print('=' * 60)
    print('AURA P0 Pipeline Test')
    print('=' * 60)

    try:
        test_audio_features()
        test_synthetic_audio()
        test_corridor_env()
        data_dir = test_data_generation()

        try:
            test_rssm_shapes()
            test_full_pipeline(data_dir)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

        elapsed = time.time() - start
        print('=' * 60)
        print(f'All P0 tests passed in {elapsed:.1f}s')
        print('=' * 60)
    except Exception as e:
        elapsed = time.time() - start
        print(f'\nFAILED after {elapsed:.1f}s: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
