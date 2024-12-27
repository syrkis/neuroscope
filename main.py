# %% main.py
#     neuroscope fun
# by: Noah Syrkis

# Imports
from functools import partial

from chex import dataclass
import esch
import jax.numpy as jnp
import jraph
import optax
from einops import rearrange
from jax import lax, random, value_and_grad, vmap, tree
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


def train_fn(rng, cfg, opt, data):
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
@vmap
def graph_fn(nodes, faces, bolds):  # nodes are also coords
    senders = jnp.concat([faces[:, 0], faces[:, 1], faces[:, 2]])[..., None]
    receivers = jnp.concat([faces[:, 1], faces[:, 2], faces[:, 0]])
    edges = jnp.ones_like(senders.squeeze())
    n_node, n_edge, globals = jnp.array([len(nodes)]), jnp.array([len(edges)]), None
    return nodes, senders, receivers, edges, globals, n_node, n_edge


# %%
def agg_fn(a, b, c):
    return a


def model_fn(params, g):
    nodes = g.nodes @ params.w
    nodes = tree.map(lambda x: agg_fn(x[g.senders], g.receivers, g.n_node), nodes)
    return g._replace(nodes=nodes)


@dataclass
class Params:
    w: jnp.ndarray


# %%
cfg = ns.utils.Config()
opt = optax.adamw(cfg.lr)
rng = random.PRNGKey(0)
data = ns.data.subject_fn(cfg)
args = map(jnp.array, zip(*tuple(map(partial(ns.fmri.mesh_fn, data, cfg), range(10)))))
graphs = jraph.GraphsTuple(*graph_fn(*args))
params = Params(w=random.normal(rng, (2, 7)))
tmp = vmap(partial(model_fn, params))(graphs)
tmp.nodes.shape


# %%
# model = jraph.GraphConvolution(update_node_fn=node_fn)  # , aggregate_nodes_fn=a
# dwg = esch.mesh(jnp.abs(bolds.T), coords[0][:, [1, 0]], shp="rect", path="tmp.svg")
