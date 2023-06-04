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
def train(data, config):
    """train function"""
    group = wandb.util.generate_id()
    for subject, (folds, _) in data.items():
        # do sweep here
        #sweep_id = wandb.sweep(sweep=config, entity='syrkis', project='neuroscope')
        #sweep_train = partial(train_subject, folds=folds, subject=subject, group=group, hem='lh')
        #wandb.agent(sweep_id, sweep_train, count=10)
        sweep_train = partial(train_subject, folds=folds, subject=subject, group=group, hem='rh')
        sweep_train(config)
        #wandb.agent(sweep_id, sweep_train, count=10)


def train_subject(config, folds, subject, group, hem) -> Tuple[List[hk.Params], List[hk.Params]]:
    """train function"""
    # config = wandb.config
    # TODO: parallelize using pmap or vmap
    rng = jax.random.PRNGKey(42)
    forward = hk.transform(partial(network_fn, config=config))
    loss_fn = partial(loss_fn_base, forward_fn=forward, config=config)
    params_lst = [forward.init(rng, img[:1]) for img, _, _, _ in folds]
    data = [make_fold(folds, fold) for fold in range(len(folds))]  # (train_data, val_data) list
    train_fold = partial(train_fold_fn, config=config, loss_fn=loss_fn, subject=subject, group=group)
    lh_val_corrs, rh_val_corrs = [], []
    for idx, (params, fold) in enumerate(zip(params_lst, data)):
        lh_val_corr, rh_val_corr = train_fold(params, rng, fold)
        lh_val_corrs.append(lh_val_corr)
        rh_val_corrs.append(rh_val_corr)
    return np.mean(lh_val_corrs).item() if hem == 'lh' else  np.mean(rh_val_corrs).item()


def train_fold_fn(params, rng, fold, config: Dict, loss_fn, subject, group) -> Tuple[float, float]:
    """train_fold function"""
    train_data, val_data = fold
    n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
    n_steps = config['parameters']['n_steps']['value']
    batch_size = config['parameters']['batch_size']['value']
    epochs = int(n_steps // (len(train_data[0]) // batch_size))
    for_log = {'n_params': n_params, 'epochs': epochs, 'subject': subject, 'group': group}
    wandb.init(project="neuroscope", entity='syrkis', config={**config, **for_log})

    # log horizontal algonauts baseline line
    linear_basline_metrics = algonauts_baseline(fold)
    opt_state = opt.init(params)
    for step in tqdm(range(n_steps)):
        batch = get_batch(train_data, batch_size)
        update = partial(update_fn, loss_fn=loss_fn)
        params, opt_state = update(params, rng, batch, opt_state)
        if step % (n_steps // N_EVALS) == 0:
            metrics = evaluate(params, rng, train_data, val_data, get_batch, config)
            wandb.log({**metrics, **linear_basline_metrics})  # plot is size is equally long for all n_steps. Add step number to change that
    wandb.finish()
    return metrics['lh_val_corr'], metrics['rh_val_corr']


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