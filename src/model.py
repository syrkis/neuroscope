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
rng = hk.PRNGSequence(jax.random.PRNGKey(42))
dropout = lambda x: hk.dropout(rate=config['dropout'], rng=next(rng), x=x)
# haiku identity function
identity = lambda x: x


# functions
def network_fn(img, dropout, fmri_dim):
    """network function"""
    # TODO: add dropout if training
    img_mlp = hk.Sequential([
        hk.Linear(100), jax.nn.gelu,
        hk.Linear(200), jax.nn.gelu,
        hk.Linear(400), jax.nn.gelu,
        hk.Linear(800), jax.nn.gelu,
    ])
    if dropout:
        pass # TODO: add dropout
    cat_mlp = hk.Sequential([
        hk.Linear(800), jax.nn.gelu,
        hk.Linear(400), jax.nn.gelu,
        hk.Linear(200), jax.nn.gelu,
        hk.Linear(80), jax.nn.sigmoid,
    ])
    fmri_mlp = hk.Sequential([
        hk.Linear(800), jax.nn.gelu,
        hk.Linear(1600), jax.nn.gelu,
        hk.Linear(fmri_dim),
    ])
    img = img_mlp(img)
    cat = cat_mlp(img)
    fmri = fmri_mlp(img)
    return fmri, cat


lh_train_forward = hk.without_apply_rng(hk.transform(partial(network_fn, dropout=True, fmri_dim=config['lh_size']))).apply
lh_infer_forward = hk.without_apply_rng(hk.transform(partial(network_fn, dropout=False, fmri_dim=config['lh_size']))).apply
lh_init = hk.transform(partial(network_fn, dropout=True, fmri_dim=config['lh_size'])).init

rh_train_forward = hk.without_apply_rng(hk.transform(partial(network_fn, dropout=True, fmri_dim=config['rh_size']))).apply
rh_infer_forward = hk.without_apply_rng(hk.transform(partial(network_fn, dropout=False, fmri_dim=config['rh_size']))).apply
rh_init = hk.transform(partial(network_fn, dropout=True, fmri_dim=config['rh_size'])).init

def loss_fn(fmri_hat, cat_hat, fmri, cat):
    """loss function"""
    fmri_loss = jnp.mean((fmri_hat - fmri) ** 2)
    cat_loss = jnp.mean((cat_hat - cat) ** 2)
    loss = config['alpha'] * fmri_loss + (1 - config['alpha']) * cat_loss
    return loss

def fmri_loss_fn(pred, fmri):
    """loss function"""
    return jnp.mean((pred - fmri) ** 2)

def cat_loss_fn(pred, cat):
    """loss function"""
    return jnp.mean((pred - cat) ** 2)

def train_loss_fn(params, batch, hem):
    """loss function"""
    img, cat, fmri = batch
    forward_fn = lh_train_forward if hem == 'lh' else rh_train_forward
    fmri_hat, cat_hat = forward_fn(params, img, cat)
    return fmri_loss_fn(fmri_hat, fmri) + cat_loss_fn(cat_hat, cat)

def infer_loss_fn(params, batch, hem):
    """loss function"""
    img, cat, fmri = batch
    forward_fn = lh_infer_forward if hem == 'lh' else rh_infer_forward
    fmri_hat, cat_hat = forward_fn(params, img, cat)
    return fmri_loss_fn(fmri_hat, fmri) + cat_loss_fn(cat_hat, cat)

lh_train_loss_fn = partial(train_loss_fn, hem='lh')
lh_infer_loss_fn = partial(infer_loss_fn, hem='lh')
rh_train_loss_fn = partial(train_loss_fn, hem='rh')
rh_infer_loss_fn = partial(infer_loss_fn, hem='rh')
loss_fns = {'lh': (lh_train_loss_fn, lh_infer_loss_fn),
            'rh': (rh_train_loss_fn, rh_infer_loss_fn)}
