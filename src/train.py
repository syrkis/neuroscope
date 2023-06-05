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
from src.model import loss_fn_base, network_fn
from src.eval import evaluate, algonauts_baseline


# constants
opt = optax.adamw(0.001)  # perhaps hyper param search for lr and weight decay

# globals
N_EVALS = 100

# types
Fold = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
Batch = Fold


# functions
def sweep(data, config):
    """train function"""
    for hem in ['lh', 'rh']:
        config['metric']['name'] = f'val_{hem}_corr'
        sweep_id = wandb.sweep(sweep=config, entity='syrkis', project='neuroscope')
        for subject, (folds, _) in data.items():
            train_subject(folds, subject, hem, sweep_id)


def train_subject(folds, subject, hem, sweep_id) -> Tuple[List[hk.Params], List[hk.Params]]:
    """train function"""
    # TODO: parallelize using pmap or vmap
    rng = jax.random.PRNGKey(42)
    data = [make_fold(folds, fold) for fold in range(len(folds))]  # (train_data, val_data) list
    group = f'{subject}_{hem}'
    for idx, fold in enumerate(data):
        train_fold = partial(train_fold_fn, subject=subject, group=group, rng=rng, fold=fold, idx=idx)
        wandb.agent(sweep_id, train_fold, count=4)


def train_fold_fn(rng, fold, idx, subject, group) -> Tuple[float, float]:
    """train_fold function"""
    fold_idx = idx + 1
    with wandb.init(project="neuroscope", entity='syrkis', group=group, reinit=True, id=f'{subject}_{fold_idx}') as run:
        config = wandb.config
        # add hem to config
        config['hem'] = group.split('_')[1]
        # update config log with fold numbers
        forward = hk.transform(partial(network_fn, config=config))
        loss_fn = partial(loss_fn_base, forward_fn=forward, config=config)
        params = forward.init(rng, fold[0][0])
        n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
        train_data, val_data = fold
        epochs = int(config.n_steps // (len(train_data[0]) // config.batch_size))
        wandb.config.update({'fold': idx + 1, 'subject': subject, 'group': group, 'n_params': n_params, 'epochs': epochs})
        linear_basline_metrics = algonauts_baseline(fold)  # might make sense to precompute
        opt_state = opt.init(params)
        update = jit(partial(update_fn, loss_fn=loss_fn))
        for step in tqdm(range(config.n_steps)):
            batch = get_batch(train_data, config.batch_size)
            params, opt_state = update(params, rng, batch, opt_state)
            if step % (config.n_steps // N_EVALS) == 0:
                metrics = evaluate(params, rng, train_data, val_data, get_batch, config)
                wandb.log({**metrics, **linear_basline_metrics})  # plot is size is equally long for all n_steps. Add step number to change that
        wandb.finish()
        metrics = evaluate(params, rng, train_data, val_data, get_batch, config, steps=50)
        return metrics['val_lh_corr'], metrics['val_rh_corr']


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

def update_fn(params: hk.Params, rng, batch: Batch, opt_state: optax.OptState, loss_fn) -> Tuple[hk.Params, optax.OptState]:
    grads = grad(loss_fn)(params, rng, batch)
    # adamw with weight decay
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state