# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk


# TODO: build haiku modules for the three modalities
def network_fn(img, cat, fmri):
    """network function"""
    img = image_network_fn(img, cat, fmri)
    cat = category_network_fn(img, cat, fmri)
    return fmri_network_fn(img, cat, fmri)


def fmri_network_fn(img, cat, fmri):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(256 * 2), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(fmri.shape[-1]), jax.nn.relu,
    ])
    return mlp(jnp.concatenate((img, cat), axis=1))


def image_network_fn(img):
    """network function"""
    img = img.reshape(img.shape[0], -1)
    mlp = hk.Sequential([
        hk.Linear(img.shape[-1]), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
    ])
    return mlp(img)

def category_network_fn(x):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
    ])
    return mlp(x)

init, forward = hk.without_apply_rng(hk.transform(network_fn))

def loss_fn(params, img, cat, fmri):
    """loss function"""
    pred = forward(params, img, cat, fmri)
    return jnp.mean((pred - fmri) ** 2)