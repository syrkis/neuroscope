# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
from jax import random, grad, jit, vmap


def init_params(layer_sizes, rng):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = random.normal(rng, (n_in, n_out)) * jnp.sqrt(2 / n_in)
        b = random.normal(rng, (n_out,)) * jnp.sqrt(2 / n_in)
        params.append((w, b))
    return params


def model(params, x):
    activations = x
    for w, b in params:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    return jax.nn.sigmoid(outputs)


def loss_fn(params, x, y):
    pred = model(params, x)  # batch_size x image_dim
    return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))


def predict(params, x):  
    preds = model(params, x)
    return (preds > 0.5).astype(int)