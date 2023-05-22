# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk
from src.utils import config


# functions
def network_fn(img, cat):
    """network function"""
    img = image_network_fn(img, cat)
    cat = category_network_fn(img, cat)
    return fmri_network_fn(img, cat)


def fmri_network_fn(img, cat):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(128 * 2), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
        hk.Linear(config['fmri_size']), jax.nn.relu,
    ])
    return mlp(jnp.concatenate((img, cat), axis=1))


def image_network_fn(img, cat):
    """network function"""
    img = img.reshape(img.shape[0], -1)
    mlp = hk.Sequential([
        hk.Linear(img.shape[-1]), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
    ])
    return mlp(img)

def category_network_fn(img, cat):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
    ])
    return mlp(cat)

init, forward = hk.without_apply_rng(hk.transform(network_fn))

def loss_fn(params, img, cat, fmri):
    """loss function"""
    pred = forward(params, img, cat)
    return jnp.mean((pred - fmri) ** 2)