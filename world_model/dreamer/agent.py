"""AURA World Model: encoder + cRSSM + decoder + training loop.

Wires together the neural network components and provides a clean
training interface using optax.
"""

import jax
import jax.numpy as jnp
import optax
import yaml
from pathlib import Path
from typing import Any

from world_model.dreamer.nets import (
    init_encoder, init_decoder, init_mlp,
    encoder_forward, decoder_forward, mlp,
    Params,
)
from world_model.dreamer.rssm import (
    RSSMConfig, RSSMState, RSSMParams,
    init_rssm, initial_state, observe, get_features, kl_loss,
)


def load_config(name: str = 'aura_debug') -> dict:
    """Load a named config from configs.yaml."""
    config_path = Path(__file__).parent / 'configs.yaml'
    with open(config_path) as f:
        configs = yaml.safe_load(f)
    if name not in configs:
        raise ValueError(f"Unknown config '{name}'. Available: {list(configs.keys())}")
    return configs[name]


def make_rssm_config(cfg: dict) -> RSSMConfig:
    """Create RSSMConfig from a config dict."""
    return RSSMConfig(
        deter_dim=cfg['deter_dim'],
        stoch_dim=cfg['stoch_dim'],
        classes=cfg['classes'],
        hidden_dim=cfg['hidden_dim'],
        embed_dim=cfg['embed_dim'],
        action_dim=cfg['action_dim'],
        context_dim=cfg['context_dim'],
        add_dcontext=cfg['add_dcontext'],
        add_context_prior=cfg['add_context_prior'],
        add_context_posterior=cfg['add_context_posterior'],
        free_nats=cfg['free_nats'],
        unimix=cfg['unimix'],
    )


def init_world_model(rng, cfg: dict) -> dict[str, Any]:
    """Initialize all world model parameters.

    Returns a nested dict of JAX arrays.
    """
    keys = jax.random.split(rng, 4)
    rssm_cfg = make_rssm_config(cfg)
    feat_dim = rssm_cfg.deter_dim + rssm_cfg.stoch_dim * rssm_cfg.classes

    return {
        'encoder': init_encoder(
            keys[0],
            channels=cfg['encoder_channels'],
            embed_dim=cfg['embed_dim'],
        ),
        'rssm': init_rssm(keys[1], rssm_cfg),
        'decoder': init_decoder(
            keys[2],
            embed_dim=feat_dim,
            channels=cfg['decoder_channels'],
        ),
        'reward_head': init_mlp(keys[3], [feat_dim, cfg['hidden_dim'], 1]),
    }


def world_model_loss(params: dict, rssm_cfg: RSSMConfig, cfg: dict,
                     batch: dict, rng) -> tuple[jnp.ndarray, dict]:
    """Compute world model loss on a batch.

    Expects batch with one-hot actions (B, T, action_dim) and float arrays.
    """
    B, T = batch['image'].shape[:2]

    # Encode all observations
    images_flat = batch['image'].reshape(B * T, 64, 64, 3)
    embeds_flat = encoder_forward(params['encoder'], images_flat)
    embeds = embeds_flat.reshape(B, T, -1)

    # Run RSSM observation model
    init = initial_state(rssm_cfg, B)
    posteriors, priors = observe(
        params['rssm'], rssm_cfg, init,
        batch['action'], embeds, batch['is_first'], batch['context'],
        rng,
    )

    # Get features for decoding
    features = get_features(posteriors)  # (B, T, feat_dim)

    # Decode images
    features_flat = features.reshape(B * T, -1)
    decoded_flat = decoder_forward(params['decoder'], features_flat)
    decoded = decoded_flat.reshape(B, T, 64, 64, 3)

    # Image reconstruction loss (MSE)
    image_loss = jnp.mean((decoded - batch['image']) ** 2)

    # Reward prediction loss
    reward_pred = mlp(params['reward_head'], features_flat)
    reward_pred = reward_pred.reshape(B, T)
    reward_loss = jnp.mean((reward_pred - batch['reward']) ** 2)

    # KL divergence loss
    kl = kl_loss(posteriors, priors, free_nats=rssm_cfg.free_nats)

    total = (
        image_loss
        + cfg['kl_scale'] * kl
        + cfg['reward_scale'] * reward_loss
    )

    metrics = {
        'total_loss': total,
        'image_loss': image_loss,
        'kl_loss': kl,
        'reward_loss': reward_loss,
    }

    return total, metrics


def imagine_trajectory(params: dict, rssm_cfg: RSSMConfig, init_state: RSSMState,
                       actions: jnp.ndarray, contexts: jnp.ndarray,
                       rng) -> jnp.ndarray:
    """Imagine a trajectory and decode to frames.

    Args:
        init_state: starting RSSM state
        actions: (B, T, action_dim) one-hot
        contexts: (B, T, context_dim)
        rng: PRNG key

    Returns:
        (B, T, 64, 64, 3) predicted frames in [0, 1].
    """
    from world_model.dreamer.rssm import imagine as rssm_imagine

    states = rssm_imagine(params['rssm'], rssm_cfg, init_state, actions, contexts, rng)
    features = get_features(states)
    B, T = features.shape[:2]
    features_flat = features.reshape(B * T, -1)
    decoded = decoder_forward(params['decoder'], features_flat)
    return decoded.reshape(B, T, 64, 64, 3)


def preprocess_batch(batch: dict, cfg: dict) -> dict:
    """Pre-convert batch arrays for JAX differentiation.

    One-hot encodes actions, casts is_first/reward/image to float32.
    """
    batch = {**batch}
    batch['action'] = jax.nn.one_hot(batch['action'], cfg['action_dim'])
    batch['is_first'] = batch['is_first'].astype(jnp.float32)
    batch['reward'] = batch['reward'].astype(jnp.float32)
    batch['image'] = batch['image'].astype(jnp.float32)
    return batch


class Trainer:
    """Optax-based training loop for the world model."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rssm_cfg = make_rssm_config(cfg)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(cfg['grad_clip']),
            optax.adam(cfg['lr']),
        )

    def init(self, rng) -> tuple[dict, Any]:
        """Initialize parameters and optimizer state.

        Returns:
            (params, opt_state)
        """
        params = init_world_model(rng, self.cfg)
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def train_step(self, params: dict, opt_state: Any,
                   batch: dict, rng) -> tuple[dict, Any, dict]:
        """Single training step.

        Returns:
            (new_params, new_opt_state, metrics)
        """
        batch = preprocess_batch(batch, self.cfg)

        rssm_cfg = self.rssm_cfg
        cfg = self.cfg

        def loss_fn(params):
            return world_model_loss(params, rssm_cfg, cfg, batch, rng)

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Add gradient norm to metrics
        grad_norm = optax.global_norm(grads)
        metrics['grad_norm'] = grad_norm

        return new_params, new_opt_state, metrics

    def train_epoch(self, params: dict, opt_state: Any,
                    dataset, rng) -> tuple[dict, Any, list[dict]]:
        """Train for one epoch over a dataset.

        Args:
            dataset: iterable of batch dicts
            rng: PRNG key

        Returns:
            (params, opt_state, list_of_metrics)
        """
        all_metrics = []
        for batch in dataset:
            rng, step_rng = jax.random.split(rng)
            params, opt_state, metrics = self.train_step(
                params, opt_state, batch, step_rng
            )
            all_metrics.append(jax.tree.map(float, metrics))
        return params, opt_state, all_metrics
