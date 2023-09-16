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
    layers = [hk.Linear(300), jax.nn.relu,
              hk.Linear(100), jax.nn.relu]

    fmri = hk.Sequential(layers)(fmri)

    if dropout_rate is not None:
        rng = hk.next_rng_key()
        fmri = hk.dropout(rng, dropout_rate, fmri)

    fmri = hk.Linear(CONFIG['image_size'] * CONFIG['image_size'] * 3)(fmri)
    fmri = fmri.reshape(-1, CONFIG['image_size'], CONFIG['image_size'], 3)
    return fmri


def loss_fn(params: hk.Params, rng: jnp.ndarray, fmri: jnp.ndarray, img: jnp.ndarray, dropout_rate: Optional[float] = None) -> jnp.ndarray:
    pred = apply(params, rng, fmri, dropout_rate=dropout_rate)
    loss = jnp.mean((pred - img) ** 2)
    return loss


# Create your partial function
init, apply = hk.transform(network_fn)