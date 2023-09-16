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
    fc = [hk.Linear(128), jax.nn.gelu,
              hk.Linear(256), jax.nn.gelu,
              hk.Linear(256), jax.nn.gelu,
              hk.Linear(CONFIG['image_size'] * CONFIG['image_size'] * 3)]

    # apply fc and reshape to image
    z = hk.Sequential(fc)(fmri)

    # apply dropout if training
    if dropout_rate is not None:
        rng = hk.next_rng_key()
        z = hk.dropout(rng, dropout_rate, z)

    # reshape to image
    z = z.reshape(-1, CONFIG['image_size'], CONFIG['image_size'], 3)

    # deconv (transpose conv) layers
    deconv = [hk.Conv2DTranspose(output_channels=21, kernel_shape=3, stride=2, padding='SAME'), jax.nn.gelu,
              hk.Conv2DTranspose(output_channels=64, kernel_shape=3, stride=2, padding='SAME'), jax.nn.gelu,
              hk.Conv2DTranspose(output_channels=32, kernel_shape=3, stride=2, padding='SAME'), jax.nn.gelu,
              hk.Conv2DTranspose(output_channels=3, kernel_shape=3, stride=2, padding='SAME')]

    # apply deconv
    z = hk.Sequential(deconv)(z)
    z = jax.nn.sigmoid(z)
    return z


def loss_fn(params: hk.Params, rng: jnp.ndarray, fmri: jnp.ndarray, img: jnp.ndarray, dropout_rate: Optional[float] = None) -> jnp.ndarray:
    pred = apply(params, rng, fmri, dropout_rate=dropout_rate)
    loss = jnp.mean((pred - img) ** 2)
    return loss


# Create your partial function
init, apply = hk.transform(network_fn)