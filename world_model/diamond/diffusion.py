"""EDM (Karras et al.) diffusion framework for DIAMOND.

Implements:
  - Training: sample sigma from log-normal, add noise, predict clean image,
    EDM-weighted MSE loss
  - Sampling: N-step Euler from sigma_max → sigma_min
  - EDM preconditioning (c_skip, c_out, c_in, c_noise)
"""

import torch
import torch.nn as nn

from world_model.diamond.unet import DiamondUNet


class EDMDiffusion(nn.Module):
    """EDM diffusion wrapper around DiamondUNet.

    Handles noise scheduling, preconditioning, loss computation,
    and Euler sampling.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.unet = DiamondUNet(cfg)

        # EDM parameters
        self.sigma_min = cfg.get('sigma_min', 0.002)
        self.sigma_max = cfg.get('sigma_max', 80.0)
        self.sigma_data = cfg.get('sigma_data', 0.5)
        self.sigma_offset_noise = cfg.get('sigma_offset_noise', 0.0)
        self.rho = cfg.get('rho', 7.0)
        self.P_mean = cfg.get('P_mean', -0.4)
        self.P_std = cfg.get('P_std', 1.2)

    # -------------------------------------------------------------------
    # EDM preconditioning (Table 1 in Karras et al.)
    # -------------------------------------------------------------------

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma.log() / 4.0

    def preconditioned_forward(self, x: torch.Tensor,
                                context_frames: torch.Tensor,
                                sigma: torch.Tensor,
                                audio_cond: torch.Tensor) -> torch.Tensor:
        """Apply EDM preconditioning and run U-Net.

        D_theta(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))

        Args:
            x: (B, 3, H, W) noisy image
            context_frames: (B, C*3, H, W) stacked context frames
            sigma: (B,) noise levels
            audio_cond: (B, 16) audio context

        Returns:
            (B, 3, H, W) denoised prediction
        """
        sigma_bc = sigma[:, None, None, None]  # broadcast shape

        # Paper: adjust sigma for offset noise in conditioners
        if self.sigma_offset_noise > 0:
            sigma_cond = (sigma_bc ** 2 + self.sigma_offset_noise ** 2).sqrt()
        else:
            sigma_cond = sigma_bc

        c_skip = self.c_skip(sigma_cond)
        c_out = self.c_out(sigma_cond)
        c_in = self.c_in(sigma_cond)

        # Paper: scale noisy input by c_in, context by 1/sigma_data (different!)
        scaled_x = c_in * x
        scaled_ctx = context_frames / self.sigma_data

        # Run U-Net with c_noise as the sigma embedding input
        F_theta = self.unet(scaled_x, scaled_ctx, self.c_noise(sigma), audio_cond)

        return c_skip * x + c_out * F_theta

    # -------------------------------------------------------------------
    # Training loss
    # -------------------------------------------------------------------

    def training_loss(self, target: torch.Tensor,
                      context_frames: torch.Tensor,
                      audio_cond: torch.Tensor) -> torch.Tensor:
        """Compute EDM training loss.

        Args:
            target: (B, 3, H, W) clean target frame in [-1, 1]
            context_frames: (B, C*3, H, W) stacked context
            audio_cond: (B, 16)

        Returns:
            Scalar loss
        """
        B = target.shape[0]
        device = target.device

        # Sample sigma from log-normal distribution, clamped to [sigma_min, sigma_max]
        log_sigma = torch.randn(B, device=device) * self.P_std + self.P_mean
        sigma = log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

        # Add noise (with optional offset noise for brightness consistency)
        noise = torch.randn_like(target)
        offset_noise_std = self.cfg.get('sigma_offset_noise', 0.0)
        if offset_noise_std > 0:
            # Offset noise: add a per-channel, spatially-constant offset
            noise = noise + offset_noise_std * torch.randn(B, 3, 1, 1, device=device)
        noisy = target + sigma[:, None, None, None] * noise

        # Predict clean image
        denoised = self.preconditioned_forward(noisy, context_frames, sigma,
                                                audio_cond)

        # EDM loss weighting: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        weight = weight[:, None, None, None]

        loss = weight * (denoised - target) ** 2
        return loss.mean()

    # -------------------------------------------------------------------
    # Euler sampler
    # -------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, context_frames: torch.Tensor,
               audio_cond: torch.Tensor,
               num_steps: int | None = None) -> torch.Tensor:
        """Generate one frame using Euler sampling.

        Args:
            context_frames: (B, C*3, H, W) stacked context
            audio_cond: (B, 16)
            num_steps: denoising steps (default from config)

        Returns:
            (B, 3, H, W) denoised frame in [-1, 1]
        """
        if num_steps is None:
            num_steps = self.cfg.get('denoising_steps', 3)

        B = context_frames.shape[0]
        H = W = self.cfg['image_size']
        device = context_frames.device

        # Build sigma schedule (EDM: geometric spacing in sigma^(1/rho))
        # Use inference sigma_max (lower than training sigma_max per DIAMOND paper)
        rho = self.rho
        sigma_max_sample = self.cfg.get('sigma_max_inference', self.sigma_max)
        sigma_max_inv = sigma_max_sample ** (1 / rho)
        sigma_min_inv = self.sigma_min ** (1 / rho)
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        sigmas = (sigma_max_inv + steps * (sigma_min_inv - sigma_max_inv)) ** rho

        # Start from pure noise at sigma_max
        x = torch.randn(B, 3, H, W, device=device) * sigmas[0]

        # Euler steps
        for i in range(num_steps):
            sigma_cur = sigmas[i].expand(B)
            sigma_next = sigmas[i + 1].expand(B)

            # Denoise
            denoised = self.preconditioned_forward(x, context_frames,
                                                    sigma_cur, audio_cond)

            # Euler step: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * (x_i - D(x_i)) / sigma_i
            d = (x - denoised) / sigma_cur[:, None, None, None]
            dt = sigma_next - sigma_cur
            x = x + d * dt[:, None, None, None]

        return x.clamp(-1, 1)
