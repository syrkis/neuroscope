# model.py
#    neuroscope model stuff
# by: Noah Syrkis

# Imports
from jax import lax, nn, random
import jax.numpy as jnp
from jaxtyping import Array
from functools import reduce
from einops import rearrange
from neuroscope import utils


# %% Constants
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


def encode_fn(params: utils.Module, x):
    x = reduce(lambda w, x: nn.gelu(conv_fn(w, x)), params.cnn, x)
    x = rearrange(x, "b c h w -> b (c h w)")
    x = reduce(lambda w, x: nn.gelu(jnp.matmul(w, x)), params.mlp, x)
    return x


def decode_fn(params: utils.Module, z):
    z = reduce(lambda w, x: nn.gelu(jnp.matmul(w, x)), params.mlp, z)
    c, h = params.cnn[0].shape[1], int((z.shape[1] // params.cnn[0].shape[1]) ** 0.5)
    z = rearrange(z, "b (c h w) -> b c h w", c=c, h=h)
    z = reduce(lambda w, x: nn.gelu(deconv_fn(w, x)), params.cnn[:-1], z)
    return nn.sigmoid(deconv_fn(z, params.cnn[-1]))


def apply_fn(params: utils.Params, x):  # apply model (assumes strucutre)
    x = encode_fn(params.encode, x)
    x = decode_fn(params.decode, x)
    return x


def module_fn(rng, mlp_d, cnn_d):  # initialize model (assumes strucutre)
    init = nn.initializers.he_uniform()
    mlp_keys, cnn_keys = (random.split(rng, len(mlp_d)), random.split(rng, len(cnn_d)))
    mlp = [init(k, (i, o)) for i, o, k in zip(mlp_d[:-1], mlp_d[1:], mlp_keys)]
    cnn = [init(k, (o, i, 3, 3)) for i, o, k in zip(cnn_d[:-1], cnn_d[1:], cnn_keys)]
    return utils.Module(mlp=mlp, cnn=cnn)


def init_fn(rng, cfg):  # initialize model (assumes strucutre)
    encode = module_fn(rng, cfg.mlp_d, cfg.cnn_d)
    decode = module_fn(rng, cfg.mlp_d[::-1], cfg.cnn_d[::-1])
    return utils.Params(encode=encode, decode=decode)
