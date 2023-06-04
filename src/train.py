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
import wandb
from typing import List, Tuple, Dict
from functools import partial
from tqdm import tqdm
from src.model import forward, loss_fn
from src.eval import evaluate, save_best_model, algonauts_baseline
from src.utils import config


# constants
opt = optax.adamw(config['lr'], weight_decay=config['weight_decay'])

# globals
N_EVALS = 100

# types
Fold = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
Batch = Fold


# functions
def train(data, config):
    """train function"""
    config['group_name'] = wandb.util.generate_id()
    for subject, (folds, _) in data.items():
        # do sweep here
        config['subject'] = subject
        sweep_id = wandb.sweep(config['sweep'], entity='syrkis', project='neuroscope', group=config['group_name'])
        sweep_train = partial(train_subject, folds=folds, config=config)
        params_lst = train_subject(folds, config)
    return params_lst


def train_subject(folds: List[Fold], config: Dict, sweep_config) -> Tuple[List[hk.Params], List[hk.Params]]:
    """train function"""
    # TODO: parallelize using pmap or vmap
    rng = jax.random.PRNGKey(42)
    params_lst = [forward.init(rng, img[:1]) for img, _, _, _ in folds]
    data = [make_fold(folds, fold) for fold in range(len(folds))]  # (train_data, val_data) list
    train_fold = partial(train_fold_fn, config=config, sweep_config=sweep_config)
    for idx, (params, fold) in enumerate(zip(params_lst, data)):
        params_lst[idx] = train_fold(params, rng, fold)
    return params_lst


def train_fold_fn(params, rng, fold, config: Dict, sweep_config) -> hk.Params:
    """train_fold function"""
    train_data, val_data = fold
    best_lh_val_loss = np.inf
    best_rh_val_loss = np.inf
    config['n_params'] = sum([p.size for p in jax.tree_util.tree_leaves(params)])
    config['epochs'] = int(config['n_steps'] // (len(train_data[0]) // config['batch_size']))
    wandb.init(project="neuroscope", entity='syrkis', config=config, group=config['group_name'])
    # log horizontal algonauts baseline line
    linear_basline_metrics = algonauts_baseline(fold)
    opt_state = opt.init(params)
    for step in tqdm(range(sweep_config['n_steps'])):
        batch = get_batch(train_data, sweep_config['batch_size'])
        params, opt_state = update(params, rng, batch, opt_state)
        if step % (sweep_config['n_steps'] // N_EVALS) == 0:
            metrics = evaluate(params, rng, train_data, val_data, get_batch, config)
            wandb.log({**metrics, **linear_basline_metrics})  # plot is size is equally long for all n_steps. Add step number to change that
    wandb.finish()
    return params


def get_batch(fold: Fold, batch_size: int) -> Batch:
    """get a batch from a split"""
    img, lh, rh, cat = fold
    idx = np.random.randint(0, img.shape[0], batch_size)
    return img[idx], lh[idx], rh[idx], cat[idx]


def make_fold(folds: List[Fold], fold: int) -> Batch:  # TODO: make sure it is correct
    """make a fold from a list of folds"""
    train_imgs = [f[0] for f in folds[:fold] + folds[fold + 1:]]
    train_lh = [f[1] for f in folds[:fold] + folds[fold + 1:]]
    train_rh = [f[2] for f in folds[:fold] + folds[fold + 1:]]
    train_cats = [f[3] for f in folds[:fold] + folds[fold + 1:]]
    train_data = tuple(map(jnp.concatenate, [train_imgs, train_lh, train_rh, train_cats]))
    return train_data, folds[fold]


@jit
def update(params: hk.Params, rng, batch: Batch, opt_state: optax.OptState) -> Tuple[hk.Params, optax.OptState]:
    grads = grad(loss_fn)(params, rng, batch)
    # adamw with weight decay
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state