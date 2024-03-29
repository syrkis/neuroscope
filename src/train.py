# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import grad, jit
import jax.numpy as jnp
import numpy as np
import wandb
import haiku as hk
import optax
from typing import List, Tuple, Dict
from functools import partial
from tqdm import tqdm
from src.model import loss_fn, init, apply


# functions
def hyperparam_fn():  # TODO: perhaps have hyperparam ranges be in config.yaml
    return {
        'batch_size': 32,
        'n_steps': 10000,
        'dropout_rate': 0.5
    }

def update_fn(params, rng, fmri, img, opt_state, opt, dropout_rate):
    rng, key = jax.random.split(rng)
    grads = grad(loss_fn)(params, key, fmri, img, dropout_rate=dropout_rate)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def train_loop(rng, opt, train_loader, val_loader, plot_batch, hyperparams):
    fold_metrics = []
    rng, key = jax.random.split(rng, 2)
    lh, rh, img = next(train_loader)
    params = init(key, lh)
    opt_state = opt.init(params)
    update = partial(update_fn, opt=opt, dropout_rate=hyperparams['dropout_rate'])
    for step in range(hyperparams['n_steps']):
        rng, key = jax.random.split(rng)
        lh, rh, img = next(train_loader)
        params, opt_state = update(params, key, lh, img, opt_state)
        if (step % (hyperparams['n_steps'] // 100)) == 0:
            rng, key = jax.random.split(rng)
            metrics = evaluate(params, key, train_loader, val_loader)
            fold_metrics.append(metrics)
    return metrics, params


def evaluate(params, rng, train_loader, val_loader, n_steps=2):
    # each batch is a tuple(lh, rh, img). Connect n_steps batches into 1
    train_loss, val_loss = 0, 0
    for _ in range(n_steps):
        rng, key_train, key_val = jax.random.split(rng, 3)
        lh, rh, img = next(train_loader)
        train_loss += loss_fn(params, key_train, lh, img)
        lh, rh, img = next(val_loader)
        val_loss += loss_fn(params, key_val, lh, img)
    train_loss /= n_steps
    val_loss /= n_steps
    return {'train_loss': train_loss, 'val_loss': val_loss}


def train_folds(kfolds, hyperparams, seed=0):
    metrics = {}
    rng = jax.random.PRNGKey(seed)
    opt = optax.adamw(learning_rate=1e-3)
    plot_batch = None
    for idx, (train_loader, val_loader) in enumerate(kfolds):
        plot_batch = next(train_loader) if plot_batch is None else plot_batch
        rng, key = jax.random.split(rng)
        fold_metrics, fold_params = train_loop(key, opt, train_loader, val_loader, plot_batch, hyperparams)
        metrics[idx] = fold_metrics
        return metrics, fold_params