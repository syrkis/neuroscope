# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk
from functools import partial


# functions
def network_fn(img, config, training=True):  # this now needs a conf (use partial)
    """network function"""
    # TODO: add dropout if training
    n_units = config['parameters']['n_units']['value']
    n_layers = config['parameters']['n_layers']['value']
    latent_dim = config['parameters']['latent_dim']['value']
    cat_size = 80
    lh_size = 19004
    rh_size = 20544
    dropout = config['parameters']['dropout']['value']
    img_mlp = hk.Sequential([
        hk.nets.MLP([n_units] * n_layers, activation=jnp.tanh),
        hk.Linear(latent_dim),
    ])
    cat_mlp = hk.Sequential([
        hk.nets.MLP([n_units] * n_layers, activation=jnp.tanh),
        hk.Linear(cat_size),
        jax.nn.sigmoid,
    ])
    lh_mlp = hk.Sequential([
        hk.nets.MLP([n_units] * n_layers, activation=jnp.tanh),
        hk.Linear(lh_size),
    ])
    rh_mlp = hk.Sequential([
        hk.nets.MLP([n_units] * n_layers, activation=jnp.tanh),
        hk.Linear(rh_size),
    ])
    img = hk.dropout(hk.next_rng_key(), dropout, img) if training else img
    z = img_mlp(img)  # get latent representation
    z = hk.dropout(hk.next_rng_key(), dropout, z) if training else z
    cat = cat_mlp(z)  # get categorical prediction
    lh = lh_mlp(z)    # get left hemisphere prediction
    rh = rh_mlp(z)    # get right hemisphere prediction
    return lh, rh, cat


# use rng
def loss_fn_base(params, rng, batch, forward_fn, config):   # this now takes a forward and conf
    """loss function"""
    beta = config['parameters']['beta']['value']
    alpha = config['parameters']['alpha']['value']
    img, lh, rh, cat = batch
    lh_hat, rh_hat, cat_hat = forward_fn.apply(params, rng, img)
    lh_loss = mse(lh_hat, lh)
    rh_loss = mse(rh_hat, rh)
    cat_loss = focal_loss(cat_hat, cat)
    fmri_loss = beta * lh_loss + (1 - beta) * rh_loss
    loss = alpha * fmri_loss + (1 - alpha) * cat_loss
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