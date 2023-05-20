# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
from jax import random
import optax
import haiku as hk


# globals
opt = optax.adam(1e-3)  # TODO: get lr from config


# TODO: build haiku modules for the three modalities
def network_fn(img, cat):
    """network function"""
    img = image_network_fn(img)
    cat = category_network_fn(cat)
    return fmri_network_fn(img, cat)


def fmri_network_fn(img, cat):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
    ])
    return mlp(jnp.concatenate((img, cat), axis=1))


def image_network_fn(x):
    """network function"""
    x = x.reshape(x.shape[0], -1)
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
    ])
    return mlp(x)

def category_network_fn(x):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
    ])
    return mlp(x)


def loss_fn(params, batch):
    """loss function"""
    img, cat, fmri = batch
    pred = network_fn(params, img, cat)
    return jnp.mean((pred - fmri) ** 2)


@jax.jit
def evaluate(params, batch):
    """evaluate function"""
    img, cat, fmri = batch
    pred = network_fn(params, img, cat)
    return jnp.mean((pred - fmri) ** 2)