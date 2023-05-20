# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import grad
import haiku as hk
import optax
import wandb
from src.model import opt, loss_fn, network_fn, evaluate

# globals
opt = optax.adam(1e-3)
init, forward = hk.without_apply_rng(hk.transform(network_fn))


# functions
def train(k_fold, config):
    """train function"""
    wandb.init(project="neuroscope", entry='syrkis', config=config)
    for fold in k_fold:
        train_loader, val_loader = fold
        params = init(jax.random.PRNGKey(42), next(train_loader))
        params = train_fold(params, train_loader, val_loader)
    wandb.finish()
        

def train_fold(params, train_loader, val_loader, state, steps=10):
    """train_fold function"""
    for step in range(steps):
        params = update(params, next(train_loader), state)
        if step % (steps // 100) == 0:
            wandb.log(step, evaluate(params, train_loader, val_loader))  # TODO: log multiple folds in one run


@jax.jit
def update(params, batch, opt_state):
    grads = grad(loss_fn)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state
