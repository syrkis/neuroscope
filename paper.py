# %% paper.py
#     neuroscope fun
# by: Noah Syrkis

# Imports
from functools import reduce

import jax.numpy as jnp
import optax
from chex import dataclass
from einops import rearrange
from jax import lax, nn, random, value_and_grad
from jax_tqdm import scan_tqdm
from jaxtyping import Array, PyTree
import esch

import neuroscope as ns


# %% Types
@dataclass
class Config:
    cnn_d = [1, 8, 16]
    mlp_d = [4096, 100]
    batch_size = 64
    image_size = 64
    stride = (2, 2)
    lr = 0.001
    subj = "subj01"
    roi = "V2d"
    hem = "lh"
    epochs = 200


stride = (2, 2)
dims = ("NCHW", "OIHW", "NCHW")
pad = "SAME"


# %% Apply functions
def conv_fn(x, kernel):
    return lax.conv(x, kernel, window_strides=stride, padding="SAME")


def deconv_fn(x: Array, kernel: Array):
    return lax.conv_transpose(
        x, kernel, strides=stride, padding=pad, dimension_numbers=dims
    )


# %% Functions
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


@dataclass
class Module:
    mlp: PyTree[jnp.ndarray]
    cnn: PyTree[jnp.ndarray]


@dataclass
class Params:
    encode: Module
    decode: Module


def encode_fn(params: Module, x):
    x = reduce(lambda w, x: nn.gelu(conv_fn(w, x)), params.cnn, x)
    x = rearrange(x, "b c h w -> b (c h w)")
    x = reduce(lambda w, x: nn.gelu(jnp.matmul(w, x)), params.mlp, x)
    return x


def decode_fn(params: Module, z):
    z = reduce(lambda w, x: nn.gelu(jnp.matmul(w, x)), params.mlp, z)
    c, h = params.cnn[0].shape[1], int((z.shape[1] // params.cnn[0].shape[1]) ** 0.5)
    z = rearrange(z, "b (c h w) -> b c h w", c=c, h=h)
    z = reduce(lambda w, x: nn.gelu(deconv_fn(w, x)), params.cnn[:-1], z)
    return nn.sigmoid(deconv_fn(z, params.cnn[-1]))


def apply_fn(params: Params, x):  # apply model (assumes strucutre)
    x = encode_fn(params.encode, x)
    x = decode_fn(params.decode, x)
    return x


def module_fn(rng, mlp_d, cnn_d):  # initialize model (assumes strucutre)
    init = nn.initializers.he_uniform()
    mlp_keys, cnn_keys = (random.split(rng, len(mlp_d)), random.split(rng, len(cnn_d)))
    mlp = [init(k, (i, o)) for i, o, k in zip(mlp_d[:-1], mlp_d[1:], mlp_keys)]
    cnn = [init(k, (o, i, 3, 3)) for i, o, k in zip(cnn_d[:-1], cnn_d[1:], cnn_keys)]
    return Module(mlp=mlp, cnn=cnn)


def init_fn(rng, cfg):  # initialize model (assumes strucutre)
    encode = module_fn(rng, cfg.mlp_d, cfg.cnn_d)
    decode = module_fn(rng, cfg.mlp_d[::-1], cfg.cnn_d[::-1])
    return Params(encode=encode, decode=decode)


def scope_fn(params, x):
    return apply_fn(params, x)


def data_fn(rng, cfg):
    data = ns.data.subject_fn(cfg.subj)
    return data


def batch_fn(key, cfg, data):
    idxs = random.permutation(key, jnp.arange(data.shape[0]))[
        : (data.shape[0] // cfg.batch_size) * cfg.batch_size
    ]
    return rearrange(data[idxs], "(s b) ... -> s b ...", b=cfg.batch_size)


def train_fn(rng, cfg, opt, data):  # using double memory
    params = init_fn(rng, cfg)
    state = (params, opt.init(params))  # type: ignore
    update = update_fn(grad_fn(apply_fn))

    @scan_tqdm(cfg.epochs)
    def aux(state, epoch_key):
        state, loss = lax.scan(update, state, batch_fn(epoch_key[1], cfg, data))
        scope = scope_fn(state[0], data[:3])
        return state, (scope, loss)

    xs = (jnp.arange(cfg.epochs), random.split(rng, cfg.epochs))
    state, (scope, loss) = lax.scan(aux, state, xs)

    return state, (scope, loss)


# %%
cfg = Config()
opt = optax.adamw(cfg.lr)
data = rearrange(ns.data.subject_fn(cfg).coco.imgs, "s w h -> s 1 w h")[:128]
rng = random.PRNGKey(0)
state, (scope, loss) = train_fn(rng, cfg, opt, data)
# %%
tmp = rearrange(scope, "t b 1 h w -> b t h w")[0]
esch.tile(tmp, animated=True, path="scope.svg", fps=10)
tmp = data[0].squeeze()
esch.tile(tmp, path="data.svg")
