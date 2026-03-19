"""Conditioned Recurrent State-Space Model (cRSSM) for AURA.

Implements the core world model dynamics with audio context conditioning,
based on the DreamerV3 RSSM architecture + context injection from
"Dreaming of Many Worlds" (cRSSM).

Context is injected at three points:
  - add_dcontext: concat context to GRU input (before deterministic state update)
  - add_context_prior: concat context to deter before computing prior logits
  - add_context_posterior: concat context to [deter, embed] before posterior logits
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any

from world_model.dreamer.nets import (
    init_linear, init_gru_cell, init_mlp,
    linear, mlp, gru_cell, Params,
)

RSSMParams = dict[str, Any]


@jax.tree_util.register_dataclass
@dataclass
class RSSMState:
    """State of the RSSM at a single timestep."""
    deter: jnp.ndarray    # (B, deter_dim) — deterministic state (GRU hidden)
    stoch: jnp.ndarray    # (B, stoch_dim, classes) — stochastic state (categorical)
    logit: jnp.ndarray    # (B, stoch_dim, classes) — logits for stochastic state


@dataclass
class RSSMConfig:
    """RSSM hyperparameters."""
    deter_dim: int = 1024
    stoch_dim: int = 32
    classes: int = 32
    hidden_dim: int = 512
    embed_dim: int = 512
    action_dim: int = 3
    context_dim: int = 16
    add_dcontext: bool = True
    add_context_prior: bool = True
    add_context_posterior: bool = True
    free_nats: float = 1.0
    unimix: float = 0.01   # uniform mixture for categorical


def init_rssm(rng, cfg: RSSMConfig) -> RSSMParams:
    """Initialize all RSSM parameters."""
    keys = jax.random.split(rng, 10)

    stoch_flat = cfg.stoch_dim * cfg.classes

    # GRU input projection: [stoch_flat + action (+ context)] → hidden → GRU
    gru_in_dim = stoch_flat + cfg.action_dim
    if cfg.add_dcontext:
        gru_in_dim += cfg.context_dim

    # Prior: deter (+ context) → hidden → stoch logits
    prior_in_dim = cfg.deter_dim
    if cfg.add_context_prior:
        prior_in_dim += cfg.context_dim

    # Posterior: [deter + embed (+ context)] → hidden → stoch logits
    post_in_dim = cfg.deter_dim + cfg.embed_dim
    if cfg.add_context_posterior:
        post_in_dim += cfg.context_dim

    return {
        'gru_input': init_mlp(keys[0], [gru_in_dim, cfg.hidden_dim]),
        'gru': init_gru_cell(keys[1], cfg.hidden_dim, cfg.deter_dim),
        'prior': init_mlp(keys[2], [prior_in_dim, cfg.hidden_dim, stoch_flat]),
        'posterior': init_mlp(keys[3], [post_in_dim, cfg.hidden_dim, stoch_flat]),
    }


def initial_state(cfg: RSSMConfig, batch_size: int) -> RSSMState:
    """Create zero-initialized RSSM state."""
    return RSSMState(
        deter=jnp.zeros((batch_size, cfg.deter_dim)),
        stoch=jnp.zeros((batch_size, cfg.stoch_dim, cfg.classes)),
        logit=jnp.zeros((batch_size, cfg.stoch_dim, cfg.classes)),
    )


def _categorical_straight_through(logit: jnp.ndarray, rng, unimix: float = 0.01):
    """Sample from categorical with straight-through gradients.

    Args:
        logit: (B, stoch_dim, classes)
        rng: JAX PRNG key
        unimix: uniform mixture ratio for exploration

    Returns:
        stoch: (B, stoch_dim, classes) one-hot with straight-through gradients
    """
    # Mix with uniform for exploration
    probs = jax.nn.softmax(logit, axis=-1)
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    probs = (1 - unimix) * probs + unimix * uniform

    # Gumbel sampling
    noise = jax.random.uniform(rng, logit.shape, minval=1e-8, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(noise))
    sample = jnp.argmax(jnp.log(probs) + gumbel, axis=-1)
    one_hot = jax.nn.one_hot(sample, probs.shape[-1])

    # Straight-through: gradients flow through probs
    stoch = one_hot + probs - jax.lax.stop_gradient(probs)
    return stoch


def img_step(params: RSSMParams, cfg: RSSMConfig, state: RSSMState,
             action: jnp.ndarray, context: jnp.ndarray,
             rng) -> RSSMState:
    """Prior / imagination step: predict next state from action + context.

    Args:
        state: current RSSM state
        action: (B, action_dim) — one-hot or continuous
        context: (B, context_dim) — audio context
        rng: PRNG key

    Returns:
        New RSSMState with prior predictions.
    """
    stoch_flat = state.stoch.reshape(state.stoch.shape[0], -1)

    # GRU input
    gru_input_parts = [stoch_flat, action]
    if cfg.add_dcontext:
        gru_input_parts.append(context)
    gru_in = jnp.concatenate(gru_input_parts, axis=-1)
    gru_in = mlp(params['gru_input'], gru_in)

    # GRU update
    deter = gru_cell(params['gru'], gru_in, state.deter)

    # Prior logits
    prior_in_parts = [deter]
    if cfg.add_context_prior:
        prior_in_parts.append(context)
    prior_in = jnp.concatenate(prior_in_parts, axis=-1)
    logit = mlp(params['prior'], prior_in)
    logit = logit.reshape(-1, cfg.stoch_dim, cfg.classes)

    stoch = _categorical_straight_through(logit, rng, cfg.unimix)
    return RSSMState(deter=deter, stoch=stoch, logit=logit)


def obs_step(params: RSSMParams, cfg: RSSMConfig, state: RSSMState,
             action: jnp.ndarray, embed: jnp.ndarray, is_first: jnp.ndarray,
             context: jnp.ndarray, rng) -> tuple[RSSMState, RSSMState]:
    """Posterior / observation step: update state using observation.

    Args:
        state: current RSSM state
        action: (B, action_dim)
        embed: (B, embed_dim) — encoded observation
        is_first: (B,) — 1.0 for first timestep (reset state)
        context: (B, context_dim)
        rng: PRNG key

    Returns:
        (posterior_state, prior_state) tuple.
    """
    k1, k2 = jax.random.split(rng)

    # Reset state at episode boundaries
    mask = (1.0 - is_first)[:, None]
    state = RSSMState(
        deter=state.deter * mask,
        stoch=state.stoch * mask[:, None],
        logit=state.logit * mask[:, None],
    )

    # Get prior (prediction before seeing observation)
    prior = img_step(params, cfg, state, action, context, k1)

    # Posterior: use observation to correct
    post_in_parts = [prior.deter, embed]
    if cfg.add_context_posterior:
        post_in_parts.append(context)
    post_in = jnp.concatenate(post_in_parts, axis=-1)
    post_logit = mlp(params['posterior'], post_in)
    post_logit = post_logit.reshape(-1, cfg.stoch_dim, cfg.classes)
    post_stoch = _categorical_straight_through(post_logit, k2, cfg.unimix)

    posterior = RSSMState(deter=prior.deter, stoch=post_stoch, logit=post_logit)
    return posterior, prior


def observe(params: RSSMParams, cfg: RSSMConfig, init_state: RSSMState,
            actions: jnp.ndarray, embeds: jnp.ndarray, is_firsts: jnp.ndarray,
            contexts: jnp.ndarray, rng) -> tuple[RSSMState, RSSMState]:
    """Run obs_step over a sequence using jax.lax.scan.

    Args:
        actions: (B, T, action_dim)
        embeds: (B, T, embed_dim)
        is_firsts: (B, T)
        contexts: (B, T, context_dim)
        rng: PRNG key

    Returns:
        (posteriors, priors) — each with fields of shape (B, T, ...).
    """
    T = actions.shape[1]
    rngs = jax.random.split(rng, T)

    def scan_fn(carry, inputs):
        state = carry
        action, embed, is_first, context, step_rng = inputs
        posterior, prior = obs_step(
            params, cfg, state, action, embed, is_first, context, step_rng
        )
        return posterior, (posterior, prior)

    # Transpose (B, T, ...) → (T, B, ...)
    scan_inputs = (
        jnp.transpose(actions, (1, 0, 2)),
        jnp.transpose(embeds, (1, 0, 2)),
        jnp.transpose(is_firsts, (1, 0)),
        jnp.transpose(contexts, (1, 0, 2)),
        rngs,
    )

    _, (posteriors, priors) = jax.lax.scan(scan_fn, init_state, scan_inputs)
    return _transpose_state(posteriors), _transpose_state(priors)


def imagine(params: RSSMParams, cfg: RSSMConfig, init_state: RSSMState,
            actions: jnp.ndarray, contexts: jnp.ndarray,
            rng) -> RSSMState:
    """Run img_step over a sequence (imagination / dreaming).

    Args:
        init_state: starting RSSM state
        actions: (B, T, action_dim)
        contexts: (B, T, context_dim)
        rng: PRNG key

    Returns:
        RSSMState with fields of shape (B, T, ...).
    """
    T = actions.shape[1]
    rngs = jax.random.split(rng, T)

    def scan_fn(carry, inputs):
        state = carry
        action, context, step_rng = inputs
        new_state = img_step(params, cfg, state, action, context, step_rng)
        return new_state, new_state

    scan_inputs = (
        jnp.transpose(actions, (1, 0, 2)),
        jnp.transpose(contexts, (1, 0, 2)),
        rngs,
    )

    _, states = jax.lax.scan(scan_fn, init_state, scan_inputs)
    return _transpose_state(states)


def _transpose_state(s: RSSMState) -> RSSMState:
    """Transpose RSSMState fields from (T, B, ...) to (B, T, ...)."""
    return RSSMState(
        deter=jnp.transpose(s.deter, (1, 0, 2)),
        stoch=jnp.transpose(s.stoch, (1, 0, 2, 3)),
        logit=jnp.transpose(s.logit, (1, 0, 2, 3)),
    )


def get_features(state: RSSMState) -> jnp.ndarray:
    """Concatenate deter and flattened stoch for downstream heads.

    Args:
        state: RSSMState with deter (B, [T,] deter_dim) and stoch (B, [T,] S, C)

    Returns:
        (B, [T,] deter_dim + stoch_dim * classes)
    """
    stoch_flat = state.stoch.reshape(*state.stoch.shape[:-2], -1)
    return jnp.concatenate([state.deter, stoch_flat], axis=-1)


def kl_loss(post: RSSMState, prior: RSSMState, free_nats: float = 1.0,
            balance: float = 0.8) -> jnp.ndarray:
    """KL divergence between posterior and prior categoricals.

    Uses free nats and KL balancing (DreamerV3 style):
    loss = balance * KL[sg(post) || prior] + (1-balance) * KL[post || sg(prior)]

    Args:
        post: posterior RSSMState
        prior: prior RSSMState
        free_nats: minimum KL (free information)
        balance: how much to push prior vs posterior

    Returns:
        Scalar KL loss.
    """
    post_probs = jax.nn.softmax(post.logit, axis=-1)
    prior_probs = jax.nn.softmax(prior.logit, axis=-1)

    def _kl(p, q):
        """KL(p || q) for categorical distributions."""
        p = jnp.clip(p, 1e-8, 1.0)
        q = jnp.clip(q, 1e-8, 1.0)
        return jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=-1)

    # KL balancing with stop gradients
    kl_prior = _kl(jax.lax.stop_gradient(post_probs), prior_probs)
    kl_post = _kl(post_probs, jax.lax.stop_gradient(prior_probs))

    # Sum over stoch_dim, mean over batch and time
    kl_prior = jnp.sum(kl_prior, axis=-1)
    kl_post = jnp.sum(kl_post, axis=-1)

    # Free nats
    kl_prior = jnp.maximum(kl_prior, free_nats)
    kl_post = jnp.maximum(kl_post, free_nats)

    kl = balance * kl_prior + (1 - balance) * kl_post
    return jnp.mean(kl)
