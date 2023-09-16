# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk
from functools import partial
from typing import Optional


# Define your function
def network_fn(fmri: jnp.ndarray, key: Optional[jnp.ndarray] = None,  dropout_rate: Optional[float] = None, image_size: int = 32) -> jnp.ndarray:
    layers = [hk.Linear(300), jax.nn.relu,
              hk.Linear(100), jax.nn.relu]

    fmri = hk.Sequential(layers)(fmri)

    if key is not None and dropout_rate is not None:
        fmri = hk.dropout(key, dropout_rate, fmri)

    fmri = hk.Linear(image_size*image_size*3)(fmri)
    fmri = fmri.reshape(-1, image_size, image_size, 3)
    return fmri


def loss_fn(params: hk.Params, rng: jnp.ndarray, fmri: jnp.ndarray, img: jnp.ndarray) -> jnp.ndarray:
    rng, key = jax.random.split(rng)
    pred = apply(params, rng, fmri, key=key, dropout_rate=0.5)
    loss = jnp.mean((pred - img) ** 2)
    return loss


# Create your partial function
init, apply = hk.transform(network_fn)