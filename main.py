# %% main.py
#     neuroscope fun
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import optax
from einops import rearrange
from jax import lax, random, value_and_grad, vmap
from jax_tqdm import scan_tqdm
from functools import partial
import jraph
import networkx as nx

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
    idxs = random.permutation(key, jnp.arange(data.shape[0]))[: (data.shape[0] // cfg.batch_size) * cfg.batch_size]
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
cfg = ns.utils.Config()
opt = optax.adamw(cfg.lr)
rng = random.PRNGKey(0)
data = ns.data.subject_fn(cfg)


# %%
@vmap
def graph_fn(coords, faces, bolds):
    nodes = coords
    senders = jnp.concat([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = jnp.concat([faces[:, 1], faces[:, 2], faces[:, 0]])
    edges = jnp.ones_like(senders)
    n_node = jnp.array([len(nodes)])
    n_edge = jnp.array([faces.shape[0]])
    globals = jnp.array([[1, 2, 44343, 4]])
    return nodes, senders, receivers, edges, globals, n_node, n_edge


# %%
def jraph_to_networkx(graph):
    G = nx.Graph()
    G.add_nodes_from(range(graph.n_node.item()))
    G.add_edges_from(zip(graph.senders.squeeze().tolist(), graph.receivers.squeeze().tolist()))
    return G


# %%
def node_fn(node, sender, receiver, globals):
    return node


def edge_fn(edge, sender, receiver, globals):
    return edge


def global_fn(node, edge, globals):
    return globals


# %% Test
model = jraph.GraphNetwork(update_node_fn=node_fn, update_edge_fn=edge_fn, update_global_fn=global_fn)
coords, faces, bolds = map(jnp.array, zip(*tuple(map(partial(ns.fmri.mesh_fn, data, cfg), range(2)))))
args = graph_fn(coords, faces, bolds)
graphs = jraph.GraphsTuple(*args)
tmp = vmap(model)(graphs)
