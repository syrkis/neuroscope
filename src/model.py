# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk
from functools import partial
from src.utils import config


# constants
IMG_N_LAYERS = config['n_layers']['img']
IMG_N_UNITS = config['n_units']['img']
FMRI_N_LAYERS = config['n_layers']['fmri']
FMRI_N_UNITS = config['n_units']['fmri']
CAT_N_LAYERS = config['n_layers']['cat']
CAT_N_UNITS = config['n_units']['cat']
LATENT_DIM = config['latent_dim']
CAT_SIZE = config['cat_size']
LH_SIZE = config['lh_size']
RH_SIZE = config['rh_size']
ALPHA = config['alpha']
BETA = config['beta']


# functions
def network_fn(img, training=True):
    """network function"""
    # TODO: add dropout if training
    img_mlp = hk.Sequential([
        hk.nets.MLP([IMG_N_UNITS] * IMG_N_LAYERS, activation=jnp.tanh),
        hk.Linear(LATENT_DIM),
    ])
    cat_mlp = hk.Sequential([
        hk.nets.MLP([CAT_N_UNITS] * CAT_N_LAYERS, activation=jnp.tanh),
        hk.Linear(CAT_SIZE),
        jax.nn.sigmoid,
    ])
    lh_mlp = hk.Sequential([
        hk.nets.MLP([FMRI_N_UNITS] * FMRI_N_LAYERS, activation=jnp.tanh),
        hk.Linear(LH_SIZE),
    ])
    rh_mlp = hk.Sequential([
        hk.nets.MLP([FMRI_N_UNITS] * FMRI_N_LAYERS, activation=jnp.tanh),
        hk.Linear(RH_SIZE),
    ])
    img = hk.dropout(hk.next_rng_key(), config['dropout'], img) if training else img
    z = img_mlp(img)  # get latent representation
    z = hk.dropout(hk.next_rng_key(), config['dropout'], z) if training else z
    cat = cat_mlp(z)  # get categorical prediction
    lh = lh_mlp(z)    # get left hemisphere prediction
    rh = rh_mlp(z)    # get right hemisphere prediction
    return lh, rh, cat


# use rng
forward = hk.transform(network_fn)

def loss_fn(params, rng, batch):
    """loss function"""
    img, lh, rh, cat = batch
    lh_hat, rh_hat, cat_hat = forward.apply(params, rng, img)
    lh_loss = mse(lh_hat, lh)
    rh_loss = mse(rh_hat, rh)
    cat_loss = focal_loss(cat_hat, cat)
    fmri_loss = BETA * lh_loss + (1 - BETA) * rh_loss
    loss = ALPHA * fmri_loss + (1 - ALPHA) * cat_loss
    return loss

def mse(pred, target):
    """loss function"""
    _mse = jnp.mean((pred - target) ** 2)
    return _mse

def bce(pred, target):
    """loss function"""
    _bce = jnp.mean(jnp.sum(-target * jnp.log(pred) - (1 - target) * jnp.log(1 - pred), axis=1))
    return _bce

def focal_loss(pred, target):
    """loss function"""
    _focal_loss = jnp.mean(jnp.sum(-target * (1 - pred) ** 2 * jnp.log(pred) - (1 - target) * pred ** 2 * jnp.log(1 - pred), axis=1))
    return _focal_loss

def soft_f1(pred, target):
    """loss function"""
    _soft_f1 = jnp.mean(jnp.sum(2 * pred * target / (pred + target), axis=1))
    return _soft_f1

def l2_reg(params):
    """loss function"""
    _l2_reg = jnp.sum(jnp.array([jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params)]))
    return _l2_reg