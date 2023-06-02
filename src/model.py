# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
import haiku as hk
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

# rng
rng = hk.PRNGSequence(jax.random.PRNGKey(42))


# functions
def network_fn(img):
    """network function"""
    # TODO: add dropout if training
    img_mlp = hk.Sequential([
        hk.nets.MLP([IMG_N_UNITS] * IMG_N_LAYERS),
        hk.Linear(LATENT_DIM),
    ])
    cat_mlp = hk.Sequential([
        hk.nets.MLP([CAT_N_UNITS] * CAT_N_LAYERS),
        hk.Linear(CAT_SIZE),
        jax.nn.sigmoid,
    ])
    lh_mlp = hk.Sequential([
        hk.nets.MLP([FMRI_N_UNITS] * FMRI_N_LAYERS),
        hk.Linear(LH_SIZE),
    ])
    rh_mlp = hk.Sequential([
        hk.nets.MLP([FMRI_N_UNITS] * FMRI_N_LAYERS),
        hk.Linear(RH_SIZE),
    ])
    z = img_mlp(img)  # get latent representation
    cat = cat_mlp(z)  # get categorical prediction
    lh = lh_mlp(z)    # get left hemisphere prediction
    rh = rh_mlp(z)    # get right hemisphere prediction
    return lh, rh, cat


init, forward = hk.without_apply_rng(hk.transform(network_fn))

def loss_fn(params, batch):
    """loss function"""
    img, lh, rh, cat = batch
    lh_hat, rh_hat, cat_hat = forward(params, img)
    lh_loss = mse(lh_hat, lh)
    rh_loss = mse(rh_hat, rh)
    cat_loss = bce(cat_hat, cat)
    return lh_loss + rh_loss + cat_loss  # TODO: add weights to losses

def mse(pred, target):
    """loss function"""
    return jnp.mean((pred - target) ** 2)

def bce(pred, target):
    """loss function"""
    return jnp.mean(jnp.where(target == 1, -jnp.log(pred), -jnp.log(1 - pred)))
