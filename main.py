# %% main.py
#     neuroscope fun
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import optax
from einops import rearrange
from jax import lax, random, value_and_grad
from jax_tqdm import scan_tqdm

import neuroscope as ns


# %% Functions
def scope_fn(params, x):
    return ns.model.apply_fn(params, x)


def grad_fn(apply_fn):
    @value_and_grad
    def aux(params, x):
        x_hat = apply_fn(params, x)
        return jnp.mean(jnp.abs(x - x_hat))

    return aux


def update_fn(grad):
    # @jit
    def aux(state, x):
        params, opt_state = state
        loss, grads = grad(params, x)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    return aux


def batch_fn(key, cfg, data):
    idxs = random.permutation(key, jnp.arange(data.shape[0]))[
        : (data.shape[0] // cfg.batch_size) * cfg.batch_size
    ]
    return rearrange(data[idxs], "(s b) ... -> s b ...", b=cfg.batch_size)


def train_fn(rng, cfg, opt, data):  # using double memory
    params = ns.model.init_fn(rng, cfg)
    state = (params, opt.init(params))  # type: ignore
    update = update_fn(grad_fn(ns.model.apply_fn))

    @scan_tqdm(cfg.epochs)
    def aux(state, epoch_key):
        state, loss = lax.scan(update, state, batch_fn(epoch_key[1], cfg, data))
        scope = scope_fn(state[0], data[:3])
        return state, (scope, loss)

    xs = (jnp.arange(cfg.epochs), random.split(rng, cfg.epochs))
    state, (scope, loss) = lax.scan(aux, state, xs)

    return state, (scope, loss)


# %%
cfg = ns.utils.Config()
opt = optax.adamw(cfg.lr)
rng = random.PRNGKey(0)
data = ns.data.subject_fn(cfg)
