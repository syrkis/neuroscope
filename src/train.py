# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import grad, pmap, vmap
import haiku as hk
import optax
import wandb
from tqdm import tqdm
from src.model import loss_fn, init
from src.eval import evaluate

# globals
opt = optax.adam(1e-3)
rng = hk.PRNGSequence(jax.random.PRNGKey(42))
steps = 1000

# functions
# TODO: parallelize using pmap and vmap
def train(k_fold: tuple, config: dict) -> list:
    """train function"""
    config['group_name'] = wandb.util.generate_id()
    loaders_lst = [next(k_fold) for _ in range(5)]  # tuple of train and val loaders
    params_lst = parallel_train_fold(loaders_lst)  # , config, steps=steps)
    return params_lst


def parallel_train_fold(loaders_lst: list) -> list: # , config, steps=1000):
    train_loaders, val_loaders = zip(*loaders_lst)
    params_lst = [init(next(rng), *next(train_loaders[i])[:2]) for i in range(5)]
    opt_state_lst = [opt.init(params) for params in params_lst]
    def train_fold(params, train_loader, val_loader, opt_state):  # , config, steps=1000):
        """train_fold function"""
        wandb.init(project="neuroscope", entity='syrkis', group='test')  #, config=config, group=config['group_name'])
        pbar = tqdm(range(steps))
        for step in pbar:
            img, cat, _, _, fmri = next(train_loader)
            params, opt_state = update(params, img, cat, fmri, opt_state)
            if step % (steps // 100) == 0:
                metrics = evaluate(params, train_loader, val_loader)
                wandb.log(metrics, step=step)
        wandb.finish()
        return params
    return pmap(train_fold, in_axes=(0, 0, 0, 0))(params_lst, train_loaders, val_loaders, opt_state_lst)  # , config, steps=steps)


@jax.jit
def update(params, img, cat, fmri, opt_state):
    grads = grad(loss_fn)(params, img, cat, fmri)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state
