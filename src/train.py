# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import grad
import haiku as hk
import optax
import wandb
from src.model import loss_fn, init
from src.eval import evaluate

# globals
opt = optax.adam(1e-3)
rng = hk.PRNGSequence(42)


# functions
def train(k_fold, config):
    """train function"""
    # wandb.init(project="neuroscope", entity='syrkis', config=config)
    for fold in k_fold:
        train_loader, val_loader = fold
        img, cat, _, _, fmri = next(train_loader)
        params = init(next(rng), img, cat, fmri)
        opt_state = opt.init(params)
        params = train_fold(params, train_loader, val_loader, opt_state)
    # wandb.finish()
    return params
        

def train_fold(params, train_loader, val_loader, opt_state, steps=10):
    """train_fold function"""
    for step in range(steps):
        img, cat, sup, cap, fmri = next(train_loader)
        params = update(params, img, cat, fmri, opt_state)
        if step % (steps // 100) == 0:
            wandb.log(step, evaluate(params, train_loader, val_loader))  # TODO: log multiple folds in one run
    return params


def update(params, img, cat, fmri, opt_state):
    grads = grad(loss_fn)(params, img, cat, fmri)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state
