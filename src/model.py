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
        hk.Linear(config['hidden_size'] * 2), jax.nn.gelu,
        hk.Linear(config['hidden_size']), jax.nn.gelu,
        hk.Linear(config['fmri_size'])
    ])
    return mlp(jnp.concatenate((img, cat), axis=1))


def image_network_fn(img, cat):
    """network function"""
    cnn = hk.Sequential([
        hk.Conv2D(16, kernel_shape=3, stride=2, padding="SAME"), jax.nn.gelu,
        hk.Conv2D(32, kernel_shape=3, stride=2, padding="SAME"), jax.nn.gelu,
        hk.Conv2D(64, kernel_shape=3, stride=2, padding="SAME"), jax.nn.gelu,
        hk.Conv2D(128, kernel_shape=3, stride=2, padding="SAME"), jax.nn.gelu,
        hk.Flatten(),
    ])
    mlp = hk.Sequential([
        hk.Linear(config['hidden_size']), jax.nn.gelu,
        hk.Linear(config['hidden_size']), jax.nn.gelu,
    ])
    return mlp(cnn(img))

def category_network_fn(img, cat):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(config['hidden_size']), jax.nn.gelu,
        hk.Linear(config['hidden_size']), jax.nn.gelu,
    ])
    return mlp(cat)

init, forward = hk.without_apply_rng(hk.transform(network_fn))

def loss_fn(params, img, cat, fmri):
    """loss function"""
    pred = forward(params, img, cat)
    return jnp.mean((pred - fmri) ** 2)