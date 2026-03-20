"""Shared utilities for DIAMOND world model."""

import torch


def get_device(device_str: str = 'auto') -> torch.device:
    """Auto-detect or create specified PyTorch device.

    Args:
        device_str: 'auto', 'cpu', 'cuda', or 'mps'.

    Returns:
        torch.device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


def quantize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Quantize [-1, 1] float tensor to uint8 and back to [-1, 1].

    Prevents floating-point drift during autoregressive generation.

    Args:
        x: (B, 3, H, W) tensor in [-1, 1].

    Returns:
        (B, 3, H, W) quantized tensor in [-1, 1].
    """
    uint8 = (x.clamp(-1, 1) * 127.5 + 127.5).round().clamp(0, 255)
    return uint8 / 127.5 - 1.0


def quantize_to_uint8_np(x: torch.Tensor) -> torch.Tensor:
    """Quantize [-1, 1] float tensor to uint8 byte tensor.

    Args:
        x: (B, 3, H, W) tensor in [-1, 1].

    Returns:
        (B, 3, H, W) uint8 byte tensor in [0, 255].
    """
    return (x.clamp(-1, 1) * 127.5 + 127.5).round().clamp(0, 255).byte()
