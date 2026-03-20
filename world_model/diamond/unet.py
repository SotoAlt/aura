"""DIAMOND U-Net with AdaGroupNorm audio conditioning.

Architecture:
  Input: 15 channels (4×3 context frames + 3 noisy target)
  Encoder: 4 levels with ResBlocks + downsampling
  Self-attention at specified resolutions
  Audio conditioning: 16-float → MLP(256) → AdaGroupNorm(scale, shift)
  Noise level: sinusoidal embedding added to conditioning
  Output: 3 channels (denoised frame prediction)
  ~4M parameters with default aura_diamond config
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for noise level sigma."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: (B,) → (B, dim)"""
        args = sigma[:, None] * self.freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaGroupNorm(nn.Module):
    """Group normalization with adaptive scale/shift from conditioning vector.

    Replaces standard GroupNorm in ResBlocks to inject audio + noise info.
    """

    def __init__(self, num_channels: int, num_groups: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W), cond: (B, cond_dim) → (B, C, H, W)"""
        x = self.norm(x)
        scale_shift = self.proj(cond)[:, :, None, None]  # (B, 2C, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """Residual block with AdaGroupNorm conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides both in_ch and out_ch
        num_groups = min(num_groups, in_ch, out_ch)
        while in_ch % num_groups != 0 or out_ch % num_groups != 0:
            num_groups -= 1

        self.norm1 = AdaGroupNorm(in_ch, num_groups, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_ch, num_groups, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, heads, C//heads, HW)
        # Scaled dot-product attention
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, C//heads)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, heads, HW, C//heads)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(attn)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class DiamondUNet(nn.Module):
    """U-Net denoiser for DIAMOND diffusion world model.

    Inputs:
        noisy_target: (B, 3, H, W) — noisy next frame
        context_frames: (B, context_frames*3, H, W) — previous frames stacked
        sigma: (B,) — noise level
        audio_cond: (B, cond_dim) — 16-float audio context vector

    Output:
        (B, 3, H, W) — denoised frame prediction (D_theta in EDM)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        image_size = cfg['image_size']
        channels = cfg['channels']       # e.g. [64, 128, 256, 256]
        num_res = cfg['num_res_blocks']   # e.g. 2
        attn_at = cfg.get('attention_at', [8])  # resolutions for self-attention
        cond_dim_in = cfg['cond_dim']     # 16
        context_frames = cfg['context_frames']  # 4

        # Input: context_frames * 3 (context) + 3 (noisy target) = 15 channels
        in_ch = context_frames * 3 + 3

        # Conditioning MLP: audio(16) + sigma_embed(256) → hidden(256)
        sigma_dim = 256
        self.sigma_embed = SinusoidalEmbedding(sigma_dim)
        cond_hidden = 256
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim_in + sigma_dim, cond_hidden),
            nn.SiLU(),
            nn.Linear(cond_hidden, cond_hidden),
        )

        # Initial projection
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = channels[0]
        res = image_size
        self._enc_channels = [ch_in]  # track for skip connections

        for level, ch_out in enumerate(channels):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res):
                blocks.append(ResBlock(ch_in, ch_out, cond_hidden))
                if res in attn_at:
                    attns.append(SelfAttention(ch_out))
                else:
                    attns.append(nn.Identity())
                ch_in = ch_out
                self._enc_channels.append(ch_in)
            self.encoder_blocks.append(blocks)
            self.encoder_attns.append(attns)

            if level < len(channels) - 1:
                self.downsamples.append(Downsample(ch_in))
                res //= 2
                self._enc_channels.append(ch_in)
            else:
                self.downsamples.append(None)

        # Middle
        self.mid_block1 = ResBlock(ch_in, ch_in, cond_hidden)
        self.mid_attn = SelfAttention(ch_in)
        self.mid_block2 = ResBlock(ch_in, ch_in, cond_hidden)

        # Decoder (reverse order, with skip connections)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(len(channels))):
            ch_out = channels[level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()

            for i in range(num_res + 1):  # +1 for the skip connection block
                skip_ch = self._enc_channels.pop()
                blocks.append(ResBlock(ch_in + skip_ch, ch_out, cond_hidden))
                if res in attn_at:
                    attns.append(SelfAttention(ch_out))
                else:
                    attns.append(nn.Identity())
                ch_in = ch_out

            self.decoder_blocks.append(blocks)
            self.decoder_attns.append(attns)

            if level > 0:
                self.upsamples.append(Upsample(ch_in))
                res *= 2
            else:
                self.upsamples.append(None)

        # Output
        num_groups = min(8, ch_in)
        while ch_in % num_groups != 0:
            num_groups -= 1
        self.out_norm = nn.GroupNorm(num_groups, ch_in)
        self.out_conv = nn.Conv2d(ch_in, 3, 3, padding=1)

    def forward(self, noisy_target: torch.Tensor, context_frames: torch.Tensor,
                sigma: torch.Tensor, audio_cond: torch.Tensor) -> torch.Tensor:
        # Build conditioning vector
        sigma_emb = self.sigma_embed(sigma)          # (B, 256)
        cond = self.cond_mlp(torch.cat([audio_cond, sigma_emb], dim=1))  # (B, 256)

        # Concatenate input: context frames + noisy target
        x = torch.cat([context_frames, noisy_target], dim=1)  # (B, 15, H, W)
        x = self.input_conv(x)

        # Encoder with skip connections
        skips = [x]
        for level in range(len(self.cfg['channels'])):
            for block, attn in zip(self.encoder_blocks[level],
                                   self.encoder_attns[level]):
                x = block(x, cond)
                x = attn(x)
                skips.append(x)
            if self.downsamples[level] is not None:
                x = self.downsamples[level](x)
                skips.append(x)

        # Middle
        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        # Decoder with skip connections
        for level, (blocks, attns) in enumerate(zip(self.decoder_blocks,
                                                     self.decoder_attns)):
            for block, attn in zip(blocks, attns):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, cond)
                x = attn(x)
            if self.upsamples[level] is not None:
                x = self.upsamples[level](x)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x
