# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk
from functools import partial
from src.utils import config


# functions
def network_fn(img, cat, is_training):
    """network function"""
    img = image_network_fn(img, is_training)
    cat = category_network_fn(cat, is_training)
    return fmri_network_fn(img, cat, is_training)


def fmri_network_fn(img, cat, is_training):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(config['hidden_size'] * 2), jax.nn.gelu,
        hk.Linear(config['hidden_size']), jax.nn.gelu,
        hk.dropout(config['dropout']) if is_training else hk.Identity(),
        hk.Linear(config['fmri_size'])
    ])
    return mlp(jnp.concatenate((img, cat), axis=1))


def image_network_fn(img, is_training):
    """network function"""
    cnn = hk.Sequential([
        hk.Conv2D(16, kernel_shape=3, stride=2, padding="SAME"), jax.nn.gelu,
        hk.Conv2D(32, kernel_shape=3, stride=2, padding="SAME"), jax.nn.gelu,
        hk.Conv2D(64, kernel_shape=3, stride=2, padding="SAME"), jax.nn.gelu,
        hk.dropout(config['dropout']) if is_training else hk.Identity(),
        hk.Conv2D(128, kernel_shape=3, stride=2, padding="SAME"),
        hk.Flatten(),
    ])
    mlp = hk.Sequential([
        hk.Linear(config['hidden_size']), jax.nn.gelu,
        hk.Linear(config['hidden_size']),
    ])
    return mlp(cnn(img))

def category_network_fn(cat, is_training):
    """network function"""
    mlp = hk.Sequential([
        hk.Linear(config['hidden_size']), jax.nn.gelu,
        hk.Linear(config['hidden_size']),
    ])
    return mlp(cat)

train_forward = hk.transform(partial(network_fn, is_training=True)).apply
infer_forward = hk.transform(partial(network_fn, is_training=False)).apply
init = jax.jit(train_forward.init, static_argnums=2)

def loss_fn(pred, fmri):
    """loss function"""
    return jnp.mean((pred - fmri) ** 2)

def train_loss(params, batch):
    """loss function"""
    img, cat, fmri = batch
    pred = train_forward(params, img, cat)
    return loss_fn(pred, fmri)

def infer_loss(params, batch):
    """loss function"""
    img, cat, fmri = batch
    pred = infer_forward(params, img, cat)
    return loss_fn(pred, fmri)
    