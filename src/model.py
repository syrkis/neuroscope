# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk
from typing import Optional
from functools import partial
from typing import Optional
from src.utils import CONFIG


# Define your function
def network_fn(fmri: jnp.ndarray,  dropout_rate: Optional[float] = None) -> jnp.ndarray:
    fc_layer = [
        hk.Linear(512), jax.nn.gelu,
        hk.Linear(512), jax.nn.gelu,
        hk.Linear(CONFIG['image_size'] * CONFIG['image_size'] * 3)
        ]

    conv_layer = [
        hk.Conv2D(64, kernel_shape=3, stride=1, padding='SAME'), jax.nn.gelu,
        hk.Conv2D(32, kernel_shape=3, stride=1, padding='SAME'), jax.nn.gelu,
        hk.Conv2D(32, kernel_shape=3, stride=1, padding='SAME'), jax.nn.gelu,
        hk.Conv2D(32, kernel_shape=3, stride=1, padding='SAME'), jax.nn.gelu,
        hk.Conv2D(16, kernel_shape=3, stride=1, padding='SAME'), jax.nn.gelu,
        hk.Conv2D(8, kernel_shape=3, stride=1, padding='SAME'), jax.nn.gelu,
        hk.Conv2D(3, kernel_shape=3, stride=1, padding='SAME'), jax.nn.gelu,
    ]

    # apply fc and reshape to image
    z = hk.Sequential(fc_layer)(fmri)

    # apply dropout if training
    if dropout_rate is not None:
        rng = hk.next_rng_key()
        z = hk.dropout(rng, dropout_rate, z)

    # reshape to image
    z = z.reshape(-1, CONFIG['image_size'], CONFIG['image_size'], 3)
    z = hk.Sequential(conv_layer)(z)
    z = jax.nn.sigmoid(z)
    return z


def loss_fn(params: hk.Params, rng: jnp.ndarray, fmri: jnp.ndarray, img: jnp.ndarray, dropout_rate: Optional[float] = None) -> jnp.ndarray:
    pred = apply(params, rng, fmri, dropout_rate=dropout_rate)
    loss = jnp.mean((pred - img) ** 2)
    return loss


# Create your partial function
init, apply = hk.transform(network_fn)