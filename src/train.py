# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import grad, jit
import jax.numpy as jnp
import numpy as np
import yaml
import haiku as hk
import optax
from typing import List, Tuple, Dict
from functools import partial
from tqdm import tqdm
from src.model import loss_fn, network_fn


def hyperparam_fn():
    return {
        'lr': np.random.choice([1e-3, 1e-4, 1e-5]),
        'batch_size': np.random.choice([32, 64]),
        'n_steps': np.random.randint(low=100, high=200),
        'dropout_rate': np.random.uniform(low=0.1, high=0.5),
    }

def update_fn(params, fmri, img, opt_state, rng, opt):
    rng, key = jax.random.split(rng)
    grads = grad(loss_fn)(params, key, fmri, img)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def train_loop(opt, init, train_loader, val_loader, plot_batch, hyperparams, rng):
    rng, key = jax.random.split(rng)
    lh, rh, img = next(train_loader)
    params = init(key, lh)
    opt_state = opt.init(params)
    update = partial(update_fn, opt=opt)
    for step in tqdm(range(hyperparams['n_steps'])):
        rng, key = jax.random.split(rng)
        lh, rh, img = next(train_loader)
        params, opt_state = update(params, lh, img, opt_state, key)
        if (step % (hyperparams['n_steps'] // 100)) == 0:
            evaluate(params, train_loader, val_loader)
            # plot_decodings(apply(params, key, plot_batch[0]), plot_batch[2])
    return params

def evaluate(params, train_loader, val_loader, n_steps=4):
    pass


def train_folds(kfolds, hyperparams, args, seed=0):
    init, apply = hk.transform(partial(network_fn, image_size=args.image_size))
    rng = jax.random.PRNGKey(seed)
    opt = optax.lion(hyperparams['lr'])
    plot_batch = None
    for train_loader, val_loader in kfolds:
        plot_batch = next(train_loader) if plot_batch is None else plot_batch
        rng, key = jax.random.split(rng)
        params = train_loop(opt, init, apply, train_loader, val_loader, plot_batch, hyperparams, key)