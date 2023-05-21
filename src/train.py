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
from src.model import loss_fn, init
from src.eval import evaluate

# globals
opt = optax.adam(1e-3)
rng = hk.PRNGSequence(jax.random.PRNGKey(42))
steps = 1000

# types
Fold = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
Batch = Fold

# functions
def train(folds: List[Fold], config: Dict) -> List[hk.Params]:
    """train function"""
    # TODO: parallelize using pmap and vmap
    config['group_name'] = wandb.util.generate_id()
    params_lst = [init(next(rng), folds[0][0], folds[0][1]) for _ in range(len(folds))]
    opt_state_lst = [opt.init(params) for params in params_lst]
    inputs = list(zip(params_lst, opt_state_lst))
    for fold in range(len(folds)):
        train_data, val_data = make_fold(folds, fold)
        params_lst[fold] = train_fold(inputs[fold] + (train_data, val_data), config)
    return params_lst


def train_fold(inputs: Tuple[hk.Params, optax.OptState, Fold,  Fold], config: Dict) -> hk.Params:
    """train_fold function"""
    wandb.init(project="neuroscope", entity='syrkis', config=config, group=config['group_name'])
    params, opt_state, train_data, val_data = inputs
    for step in range(steps):
        img, cat, fmri = get_batch(train_data, config['batch_size'])
        params, opt_state = update(params, img, cat, fmri, opt_state)
        if step % (steps // 100) == 0:
            metrics = evaluate(params, train_data, val_data)
            wandb.log(metrics, step=step)
    wandb.finish()
    return params


def get_batch(fold: Fold, batch_size: int) -> Batch:
    """get a batch from a split"""
    print(fold)
    exit()
    img, cat, fmri = fold
    idx = np.random.randint(0, img.shape[0], batch_size)
    return img[idx], cat[idx], fmri[idx]


def make_fold(folds: List[Fold], fold: int) -> Batch:
    """make a fold from a list of folds"""
    con_mod = lambda i, j: jnp.concatenate(folds[i][j], axis=0)
    train_data = tuple([con_mod(i, j) for i in range(len(folds)) for j in range(3) if i != fold])
    return train_data, folds[fold]


@jit
def update(params: hk.Params, img: jnp.ndarray, cat: jnp.ndarray, fmri: jnp.ndarray, opt_state: optax.OptState) -> Tuple[hk.Params, optax.OptState]:
    grads = grad(loss_fn)(params, img, cat, fmri)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state