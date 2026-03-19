"""Checkpoint save/load for AURA world model.

Uses pickle for JAX array serialization. Stores params, optimizer state,
config, and training step in a single file.
"""

import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


def save_checkpoint(path: str, params: dict, opt_state, cfg: dict, step: int):
    """Save a training checkpoint.

    Args:
        path: File path (e.g. 'checkpoints/aura-v0.1.ckpt').
        params: World model parameters (nested JAX arrays).
        opt_state: Optax optimizer state.
        cfg: Training config dict.
        step: Current training step.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert JAX arrays to numpy for portable serialization
    _to_np = lambda x: np.asarray(x) if hasattr(x, 'shape') else x
    checkpoint = {
        'params': jax.tree.map(_to_np, params),
        'opt_state': jax.tree.map(_to_np, opt_state),
        'cfg': cfg,
        'step': step,
    }

    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: str) -> dict:
    """Load a training checkpoint.

    Args:
        path: File path to checkpoint.

    Returns:
        Dict with 'params', 'opt_state', 'cfg', 'step'.
        Arrays are converted back to JAX arrays.
    """
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Convert numpy arrays back to JAX
    checkpoint['params'] = jax.tree.map(jnp.array, checkpoint['params'])
    checkpoint['opt_state'] = jax.tree.map(
        lambda x: jnp.array(x) if hasattr(x, 'shape') else x,
        checkpoint['opt_state'],
    )

    return checkpoint
