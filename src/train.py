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
from src.model import lh_init, rh_init, train_loss_fn
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
        lh_params_lst, rh_params_lst = train_subject(folds, config)
        np.save(f'results/models/{subject}_lh.npy', lh_params_lst)
        np.save(f'results/models/{subject}_rh.npy', rh_params_lst)


def train_subject(folds: List[Fold], config: Dict) -> Tuple[List[hk.Params], List[hk.Params]]:
    """train function"""
    # TODO: parallelize using pmap or vmap
    lh_params_lst = [lh_init(next(rng), img, cat) for img, cat, _ in folds]
    rh_params_lst = [rh_init(next(rng), img, cat) for img, cat, _ in folds]
    data = [make_fold(folds, fold) for fold in range(len(folds))]  # (train_data, val_data) list
    train_fold = partial(train_fold_fn, config=config)
    for idx, (lh_params, rh_params, fold) in enumerate(zip(lh_params_lst, rh_params_lst, data)):
        params = train_fold(lh_params, rh_params, fold)
        lh_params_lst[idx] = params[0]
        rh_params_lst[idx] = params[1]
    return lh_params_lst, rh_params_lst


def train_fold_fn(lh_params, rh_params, fold, config: Dict) -> hk.Params:
    """train_fold function"""
    train_data, val_data = fold
    wandb.init(project="neuroscope", entity='syrkis', config=config, group=config['group_name'])
    lh_opt_state = opt.init(lh_params)
    rh_opt_state = opt.init(rh_params)
    for step in tqdm(range(config['n_steps'])):
        lh_batch, rh_batch = get_batch(train_data, config['batch_size'])
        lh_params, lh_opt_state = update(lh_params, lh_batch, lh_opt_state)
        rh_batch, rh_opt_state = update(rh_params, rh_batch, rh_opt_state)
        if step % (config['n_steps'] // 100) == 0:
            metrics = evaluate(lh_params, rh_params, train_data, val_data, get_batch, config)
            wandb.log(metrics, step=step)
    wandb.finish()
    return lh_params, rh_params


def get_batch(fold: Fold, batch_size: int) -> Batch:
    """get a batch from a split"""
    img, cat, fmri = fold
    idx = np.random.randint(0, img.shape[0], batch_size)
    return img[idx], cat[idx], fmri[idx]


def make_fold(folds: List[Fold], fold: int) -> Batch:
    """make a fold from a list of folds"""
    train_imgs = [f[0] for f in folds[:fold] + folds[fold + 1:]]
    train_cats = [f[1] for f in folds[:fold] + folds[fold + 1:]]
    train_fmris = [f[2] for f in folds[:fold] + folds[fold + 1:]]
    train_data = tuple(map(jnp.concatenate, [train_imgs, train_cats, train_fmris]))
    return train_data, folds[fold]


@jit
def update(params: hk.Params, batch: Batch, opt_state: optax.OptState) -> Tuple[hk.Params, optax.OptState]:
    grads = grad(train_loss_fn)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state