"""DIAMOND training loop for AURA.

Supports:
  - Standard diffusion training (predict clean image from noisy)
  - Autoregressive (AR) training: after burnin, replace ground-truth context
    with model's own predictions during training
  - EMA model for evaluation
  - Checkpoint save/resume

Usage:
    python -m world_model.diamond.train --config aura_diamond_tiny --data data/matsya --steps 10
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import yaml

from world_model.diamond.diffusion import EDMDiffusion
from world_model.diamond.dataset import make_dataloader
from world_model.diamond.utils import get_device, quantize_to_uint8


def load_config(name: str) -> dict:
    """Load a named config from configs.yaml."""
    cfg_path = Path(__file__).parent / 'configs.yaml'
    with open(cfg_path) as f:
        all_configs = yaml.safe_load(f)
    if name not in all_configs:
        raise ValueError(f'Config "{name}" not found. Available: {list(all_configs.keys())}')
    return all_configs[name]


@torch.no_grad()
def update_ema(ema_model: EDMDiffusion, model: EDMDiffusion, decay: float):
    """Update EMA model parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 1 - decay)


@torch.no_grad()
def ar_training_batch(model: EDMDiffusion, batch: dict,
                      rollout_len: int, device: torch.device) -> dict:
    """Create an autoregressive training batch.

    Takes a standard batch, rolls out the model for `rollout_len` steps,
    then returns a new batch where context frames include model predictions
    (teaching the model to handle its own imperfect outputs).

    Args:
        model: Current model (eval mode used for rollout).
        batch: Standard batch from dataloader.
        rollout_len: Number of AR steps.
        device: Device.

    Returns:
        Modified batch with AR context.
    """
    model.eval()
    ctx = batch['context_frames'].to(device)  # (B, N*3, H, W)
    audio = batch['audio_context'].to(device)  # (B, 16)

    C = model.cfg['context_frames']  # number of context frames (4)

    # Start from the given context and roll forward
    # ctx has shape (B, C*3, H, W) — split into individual frames
    frames = list(ctx.chunk(C, dim=1))  # list of C tensors, each (B, 3, H, W)

    for _ in range(rollout_len):
        # Stack last C frames as context
        context_stack = torch.cat(frames[-C:], dim=1)  # (B, C*3, H, W)
        # Generate next frame
        pred = model.sample(context_stack, audio)
        # Quantize to uint8 and back to prevent drift
        pred = quantize_to_uint8(pred)
        frames.append(pred)

    model.train()

    # Use the last C frames as new context, last prediction as target
    new_ctx = torch.cat(frames[-(C + 1):-1], dim=1)  # (B, C*3, H, W)
    new_target = frames[-1]  # (B, 3, H, W)

    return {
        'context_frames': new_ctx,
        'target_frame': new_target,
        'audio_context': audio,
    }


def train(cfg_name: str, data_dir: str, total_steps: int,
          checkpoint_path: str, resume: str | None = None,
          log_every: int = 100, save_every: int = 5000,
          device_str: str = 'auto'):
    """Main training function.

    Args:
        cfg_name: Config name from configs.yaml.
        data_dir: Path to NPZ episode data.
        total_steps: Total training steps.
        checkpoint_path: Where to save checkpoints.
        resume: Optional checkpoint to resume from.
        log_every: Log interval.
        save_every: Checkpoint save interval.
        device_str: 'auto', 'cpu', 'cuda', or 'mps'.
    """
    cfg = load_config(cfg_name)

    device = get_device(device_str)
    print(f'Device: {device}')

    # Model
    model = EDMDiffusion(cfg).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {param_count:,} ({param_count / 1e6:.1f}M)')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999),
                                   weight_decay=cfg.get('weight_decay', 0.01))

    # Data
    num_workers = 0 if device.type == 'cpu' else 2
    loader = make_dataloader(data_dir, cfg, num_workers=num_workers)
    data_iter = iter(loader)

    # Resume
    start_step = 0
    if resume:
        print(f'Resuming from {resume}')
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0)
        print(f'  Resumed at step {start_step}')

    # AR training params
    ar_start = cfg.get('ar_start_step', float('inf'))
    ar_frac = cfg.get('ar_fraction', 0.0)
    ar_len = cfg.get('ar_rollout_len', 4)

    # Training loop
    print(f'\nTraining {cfg_name}: {total_steps} steps, batch_size={cfg["batch_size"]}')
    print(f'AR rollouts: start={ar_start}, fraction={ar_frac}, len={ar_len}')
    print()

    model.train()
    loss_accum = 0.0
    t_start = time.time()

    for step in range(start_step, total_steps):
        # Get batch (restart iterator if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Maybe do AR training
        use_ar = (step >= ar_start and ar_frac > 0
                  and torch.rand(1).item() < ar_frac)

        if use_ar:
            batch = ar_training_batch(model, batch, ar_len, device)
            ctx = batch['context_frames']
            target = batch['target_frame']
            audio = batch['audio_context']
        else:
            ctx = batch['context_frames'].to(device)
            target = batch['target_frame'].to(device)
            audio = batch['audio_context'].to(device)

        # Forward + loss
        loss = model.training_loss(target, ctx, audio)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if cfg.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()

        # EMA update
        update_ema(ema_model, model, cfg.get('ema_decay', 0.999))

        loss_accum += loss.item()

        # Logging
        if (step + 1) % log_every == 0:
            avg_loss = loss_accum / log_every
            elapsed = time.time() - t_start
            steps_per_sec = log_every / elapsed
            ar_tag = ' [AR]' if use_ar else ''
            print(f'step {step + 1:>6d}/{total_steps}  '
                  f'loss={avg_loss:.4f}  '
                  f'{steps_per_sec:.1f} steps/s{ar_tag}')
            loss_accum = 0.0
            t_start = time.time()

        # Save checkpoint
        if (step + 1) % save_every == 0 or step + 1 == total_steps:
            ckpt_path = Path(checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step + 1,
                'cfg': cfg,
                'cfg_name': cfg_name,
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, str(ckpt_path))
            print(f'  Checkpoint saved: {ckpt_path}')

    print(f'\nTraining complete. Final checkpoint: {checkpoint_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DIAMOND world model')
    parser.add_argument('--config', type=str, default='aura_diamond_tiny',
                        help='Config name from configs.yaml')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to NPZ episode data directory')
    parser.add_argument('--steps', type=int, default=30000,
                        help='Total training steps')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/diamond.ckpt',
                        help='Checkpoint save path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=5000)
    args = parser.parse_args()

    train(
        cfg_name=args.config,
        data_dir=args.data,
        total_steps=args.steps,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        log_every=args.log_every,
        save_every=args.save_every,
        device_str=args.device,
    )
