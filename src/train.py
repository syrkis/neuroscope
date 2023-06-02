# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import grad, jit
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import wandb
from typing import List, Tuple, Dict
from functools import partial
from tqdm import tqdm
from src.model import init, loss_fn
from src.eval import evaluate


# constants
opt = optax.adam(1e-3)
rng = hk.PRNGSequence(jax.random.PRNGKey(42))

# types
Fold = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
Batch = Fold


# functions
def train(data, config):
    """train function"""
    config['group_name'] = wandb.util.generate_id()
    for subject, (folds, _) in data.items():
        config['subject'] = subject
        params_lst = train_subject(folds, config)


def train_subject(folds: List[Fold], config: Dict) -> Tuple[List[hk.Params], List[hk.Params]]:
    """train function"""
    # TODO: parallelize using pmap or vmap
    params_lst = [init(next(rng), img[:1]) for img, _, _, _ in folds]
    config['n_params'] = sum([p.size for p in jax.tree_util.tree_leaves(params_lst[0])])
    data = [make_fold(folds, fold) for fold in range(len(folds))]  # (train_data, val_data) list
    train_fold = partial(train_fold_fn, config=config)
    for idx, (params, fold) in enumerate(zip(params_lst, data)):
        params_lst[idx] = train_fold(params, fold)
    return params_lst


def train_fold_fn(params, fold, config: Dict) -> hk.Params:
    """train_fold function"""
    train_data, val_data = fold
    wandb.init(project="neuroscope", entity='syrkis', config=config, group=config['group_name'])
    opt_state = opt.init(params)
    for step in tqdm(range(config['n_steps'])):
        batch = get_batch(train_data, config['batch_size'])
        params, opt_state = update(params, batch, opt_state)
        if step % (config['n_steps'] // 100) == 0:
            metrics = evaluate(params, train_data, val_data, get_batch, config)
            wandb.log(metrics, step=step)
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
def update(params: hk.Params, batch: Batch, opt_state: optax.OptState) -> Tuple[hk.Params, optax.OptState]:
    grads = grad(loss_fn)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state