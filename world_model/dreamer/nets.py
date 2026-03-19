"""Neural network primitives for AURA world model.

Plain JAX implementations of encoder, decoder, MLP, and GRU cell.
No framework dependencies — just jax.numpy and parameter dicts.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

Params = dict[str, Any]


# --- Parameter initialization ---

def _glorot(rng, shape):
    """Glorot uniform initialization."""
    fan_in = np.prod(shape[:-1])
    fan_out = shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(rng, shape, minval=-limit, maxval=limit)


def init_linear(rng, in_dim: int, out_dim: int) -> Params:
    return {
        'w': _glorot(rng, (in_dim, out_dim)),
        'b': jnp.zeros(out_dim),
    }


def init_conv_params(rng, in_ch: int, out_ch: int, kernel: int = 4) -> Params:
    """Initialize conv2d / conv2d_transpose params (same HWIO layout)."""
    return {
        'w': _glorot(rng, (kernel, kernel, in_ch, out_ch)),
        'b': jnp.zeros(out_ch),
    }


def init_gru_cell(rng, input_dim: int, hidden_dim: int) -> Params:
    """GRU cell: gates (reset, update) + candidate."""
    keys = jax.random.split(rng, 6)
    return {
        'Wr': _glorot(keys[0], (input_dim, hidden_dim)),
        'Ur': _glorot(keys[1], (hidden_dim, hidden_dim)),
        'br': jnp.zeros(hidden_dim),
        'Wz': _glorot(keys[2], (input_dim, hidden_dim)),
        'Uz': _glorot(keys[3], (hidden_dim, hidden_dim)),
        'bz': jnp.zeros(hidden_dim),
        'Wh': _glorot(keys[4], (input_dim, hidden_dim)),
        'Uh': _glorot(keys[5], (hidden_dim, hidden_dim)),
        'bh': jnp.zeros(hidden_dim),
    }


def init_mlp(rng, dims: list[int]) -> list[Params]:
    """Initialize a multi-layer perceptron."""
    layers = []
    for i in range(len(dims) - 1):
        rng, k = jax.random.split(rng)
        layers.append(init_linear(k, dims[i], dims[i + 1]))
    return layers


# --- Forward passes ---

def linear(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params['w'] + params['b']


def mlp(params: list[Params], x: jnp.ndarray, activation=jax.nn.elu) -> jnp.ndarray:
    """MLP forward: activation on all but last layer."""
    for i, layer in enumerate(params):
        x = linear(layer, x)
        if i < len(params) - 1:
            x = activation(x)
    return x


def conv2d(params: Params, x: jnp.ndarray, stride: int = 2) -> jnp.ndarray:
    """2D convolution: x is (B, H, W, C)."""
    out = jax.lax.conv_general_dilated(
        x, params['w'],
        window_strides=(stride, stride),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return out + params['b']


def conv2d_transpose(params: Params, x: jnp.ndarray, stride: int = 2) -> jnp.ndarray:
    """Transposed 2D convolution: x is (B, H, W, C)."""
    out = jax.lax.conv_transpose(
        x, params['w'],
        strides=(stride, stride),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return out + params['b']


def gru_cell(params: Params, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """GRU cell forward: returns new hidden state."""
    r = jax.nn.sigmoid(x @ params['Wr'] + h @ params['Ur'] + params['br'])
    z = jax.nn.sigmoid(x @ params['Wz'] + h @ params['Uz'] + params['bz'])
    h_tilde = jnp.tanh(x @ params['Wh'] + (r * h) @ params['Uh'] + params['bh'])
    h_new = (1 - z) * h + z * h_tilde
    return h_new


# --- Encoder ---

def init_encoder(rng, channels: list[int] = None, embed_dim: int = 512) -> Params:
    """CNN encoder: 64×64×3 → embed_dim.

    4 conv layers with stride 2: 64→32→16→8→4, then flatten and project.
    """
    if channels is None:
        channels = [32, 64, 128, 256]
    keys = jax.random.split(rng, len(channels) + 1)
    convs = []
    in_ch = 3
    for i, out_ch in enumerate(channels):
        convs.append(init_conv_params(keys[i], in_ch, out_ch, kernel=4))
        in_ch = out_ch
    # After 4 stride-2 convs on 64×64: 4×4×channels[-1]
    flat_dim = 4 * 4 * channels[-1]
    proj = init_linear(keys[-1], flat_dim, embed_dim)
    return {'convs': convs, 'proj': proj}


def encoder_forward(params: Params, image: jnp.ndarray) -> jnp.ndarray:
    """Encode image (B, 64, 64, 3) float [0,1] → (B, embed_dim)."""
    x = image
    for conv_params in params['convs']:
        x = conv2d(conv_params, x, stride=2)
        x = jax.nn.elu(x)
    # Flatten spatial dims
    batch = x.shape[0]
    x = x.reshape(batch, -1)
    x = linear(params['proj'], x)
    x = jax.nn.elu(x)
    return x


# --- Decoder ---

def init_decoder(rng, embed_dim: int = 512,
                 channels: list[int] = None) -> Params:
    """Transposed CNN decoder: feature_dim → 64×64×3.

    Project to 4×4×channels[0], then 4 transposed convs.
    """
    if channels is None:
        channels = [256, 128, 64, 32]
    keys = jax.random.split(rng, len(channels) + 2)

    proj = init_linear(keys[0], embed_dim, 4 * 4 * channels[0])
    deconvs = []
    for i in range(len(channels) - 1):
        deconvs.append(init_conv_params(keys[i + 1], channels[i], channels[i + 1], kernel=4))
    # Final layer outputs 3 channels (RGB)
    deconvs.append(init_conv_params(keys[-1], channels[-1], 3, kernel=4))
    return {'proj': proj, 'deconvs': deconvs}


def decoder_forward(params: Params, features: jnp.ndarray) -> jnp.ndarray:
    """Decode features (B, feat_dim) → (B, 64, 64, 3) in [0, 1]."""
    batch = features.shape[0]
    x = linear(params['proj'], features)
    x = jax.nn.elu(x)
    # Infer channels from proj weight shape: proj maps to 4*4*C, so C = out_dim / 16
    init_channels = params['proj']['w'].shape[-1] // 16
    x = x.reshape(batch, 4, 4, init_channels)

    for i, deconv_params in enumerate(params['deconvs']):
        x = conv2d_transpose(deconv_params, x, stride=2)
        if i < len(params['deconvs']) - 1:
            x = jax.nn.elu(x)
        else:
            x = jax.nn.sigmoid(x)  # output in [0, 1]
    return x
