"""AURA training script.

Works locally (CPU, aura_debug) and on Colab (GPU, aura).

Usage:
    python -m world_model.train --config aura_debug --data data/train --steps 1000
    python -m world_model.train --config aura --data data/train --steps 100000 --checkpoint checkpoints/aura-v0.1.ckpt
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from world_model.dreamer.agent import (
    Trainer, load_config, imagine_trajectory, make_rssm_config,
    world_model_loss, preprocess_batch,
)
from world_model.dreamer.rssm import initial_state
from world_model.dreamer.checkpoint import save_checkpoint, load_checkpoint
from world_model.dreamer.logging import init_wandb, log_metrics, log_frames, finish_wandb, HAS_WANDB
from world_model.data.generate import NPZDataset


def train(cfg_name: str, data_dir: str, steps: int, checkpoint_path: str,
          resume: str | None = None, log_every: int = 100,
          save_every: int = 1000, val_every: int = 500,
          val_episodes: int = 10, wandb_enabled: bool = True):
    """Main training loop."""
    cfg = load_config(cfg_name)
    print(f'Config: {cfg_name}')
    print(f'  deter_dim={cfg["deter_dim"]}, stoch_dim={cfg["stoch_dim"]}, '
          f'classes={cfg["classes"]}')
    print(f'  batch_size={cfg["batch_size"]}, seq_length={cfg["seq_length"]}, '
          f'lr={cfg["lr"]}')
    print(f'Data: {data_dir}')
    print(f'Device: {jax.devices()[0]}')

    # Init wandb
    if wandb_enabled:
        init_wandb(cfg, run_name=f'aura-{cfg_name}')

    # Init model
    trainer = Trainer(cfg)
    rng = jax.random.key(42)

    if resume:
        print(f'Resuming from {resume}')
        ckpt = load_checkpoint(resume)
        params, opt_state = ckpt['params'], ckpt['opt_state']
        start_step = ckpt['step']
    else:
        rng, init_rng = jax.random.split(rng)
        params, opt_state = trainer.init(init_rng)
        start_step = 0

    # Load dataset
    dataset = NPZDataset(data_dir, seq_length=cfg['seq_length'],
                         batch_size=cfg['batch_size'])

    # Validation dataset (last 10% of episodes by path)
    all_paths = dataset.episode_paths
    val_split = max(1, len(all_paths) // 10)
    val_dataset = NPZDataset(data_dir, seq_length=cfg['seq_length'],
                             batch_size=cfg['batch_size'])
    val_dataset.episode_paths = all_paths[-val_split:]
    dataset.episode_paths = all_paths[:-val_split]
    print(f'Train episodes: {len(dataset.episode_paths)}, '
          f'Val episodes: {len(val_dataset.episode_paths)}')

    # Training loop
    data_rng = np.random.default_rng(42)
    best_val_loss = float('inf')
    t0 = time.time()

    pbar = tqdm(range(start_step, start_step + steps), desc='Training')
    for step in pbar:
        # Sample batch
        batch_np = dataset.sample_batch(data_rng)
        batch = {k: jnp.array(v) for k, v in batch_np.items()}

        # Train step
        rng, step_rng = jax.random.split(rng)
        params, opt_state, metrics = trainer.train_step(
            params, opt_state, batch, step_rng
        )

        # Convert metrics to float
        metrics_f = {k: float(v) for k, v in metrics.items()}

        # Progress bar
        pbar.set_postfix(
            loss=f'{metrics_f["total_loss"]:.4f}',
            img=f'{metrics_f["image_loss"]:.4f}',
            kl=f'{metrics_f["kl_loss"]:.4f}',
        )

        # Log metrics
        if (step + 1) % log_every == 0:
            metrics_f['step'] = step + 1
            metrics_f['elapsed_sec'] = time.time() - t0
            metrics_f['steps_per_sec'] = (step + 1 - start_step) / (time.time() - t0)
            log_metrics(metrics_f, step + 1)
            tqdm.write(
                f'[step {step+1}] loss={metrics_f["total_loss"]:.4f} '
                f'img={metrics_f["image_loss"]:.4f} kl={metrics_f["kl_loss"]:.4f} '
                f'grad={metrics_f["grad_norm"]:.4f} '
                f'({metrics_f["steps_per_sec"]:.1f} steps/s)'
            )

        # Validation
        if (step + 1) % val_every == 0:
            val_losses = []
            val_rng = np.random.default_rng(0)
            for _ in range(val_episodes):
                val_batch_np = val_dataset.sample_batch(val_rng)
                val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}
                rng, vr = jax.random.split(rng)
                val_batch = preprocess_batch(val_batch, cfg)
                _, vm = world_model_loss(params, trainer.rssm_cfg, cfg, val_batch, vr)
                val_losses.append({k: float(v) for k, v in vm.items()})

            avg_val = {k: np.mean([m[k] for m in val_losses]) for k in val_losses[0]}
            val_metrics = {f'val_{k}': v for k, v in avg_val.items()}
            log_metrics(val_metrics, step + 1)
            tqdm.write(
                f'[val step {step+1}] loss={avg_val["total_loss"]:.4f} '
                f'img={avg_val["image_loss"]:.4f} kl={avg_val["kl_loss"]:.4f}'
            )

            if avg_val['total_loss'] < best_val_loss:
                best_val_loss = avg_val['total_loss']
                best_path = str(Path(checkpoint_path).with_suffix('.best.ckpt'))
                save_checkpoint(best_path, params, opt_state, cfg, step + 1)
                tqdm.write(f'  New best val loss: {best_val_loss:.4f} → saved {best_path}')

        # Save checkpoint
        if (step + 1) % save_every == 0:
            save_checkpoint(checkpoint_path, params, opt_state, cfg, step + 1)
            tqdm.write(f'Checkpoint saved: {checkpoint_path} (step {step+1})')

        # Log imagined frames periodically (only when wandb active)
        if wandb_enabled and HAS_WANDB and (step + 1) % save_every == 0:
            rng, img_rng = jax.random.split(rng)
            rssm_cfg = make_rssm_config(cfg)
            init_s = initial_state(rssm_cfg, 1)
            T_imagine = min(20, cfg['seq_length'])
            actions = jax.nn.one_hot(
                jnp.zeros((1, T_imagine), dtype=jnp.int32), cfg['action_dim']
            )
            contexts = jnp.ones((1, T_imagine, cfg['context_dim'])) * 0.5
            frames = imagine_trajectory(params, rssm_cfg, init_s, actions, contexts, img_rng)
            frames_np = np.array(frames[0])  # (T, H, W, 3)
            log_frames(frames_np, step + 1, 'imagined')

    # Final save
    save_checkpoint(checkpoint_path, params, opt_state, cfg, start_step + steps)
    elapsed = time.time() - t0
    print(f'\nTraining complete: {steps} steps in {elapsed:.1f}s '
          f'({steps/elapsed:.1f} steps/s)')
    print(f'Final checkpoint: {checkpoint_path}')
    finish_wandb()

    return params, opt_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AURA world model')
    parser.add_argument('--config', type=str, default='aura_debug',
                        help='Config name from configs.yaml')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.ckpt',
                        help='Checkpoint save path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    train(
        cfg_name=args.config,
        data_dir=args.data,
        steps=args.steps,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        log_every=args.log_every,
        save_every=args.save_every,
        val_every=args.val_every,
        wandb_enabled=not args.no_wandb,
    )
