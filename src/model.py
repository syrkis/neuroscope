# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
from jax import random
import haiku as hk


# TODO: build haiku modules for the three modalities
# init_params
def init_mlp(layer_sizes, rng):  # TODO: switch to haiku
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = random.normal(rng, (n_in, n_out)) * jnp.sqrt(2 / n_in)
        b = random.normal(rng, (n_out,)) * jnp.sqrt(2 / n_in)
        params.append((w, b))
    return params

 
def init_cnn(config, rng):
    params = []
    channel_sizes = [config["data"]["n_channels"]] + config["model"]["hyperparams"][
        "channel_sizes"
    ]
    kernel_sizes = list(
        zip(
            channel_sizes[:-1],
            channel_sizes[1:],
            config["model"]["hyperparams"]["kernel_sizes"],
        )
    )
    for c_in, c_out, k in kernel_sizes:
        w = random.normal(rng, (c_out, c_in, k, k)) * jnp.sqrt(2 / (k * k * c_in))
        b = random.normal(rng, (c_out,)) * jnp.sqrt(2 / (k * k * c_in))
        params.append((w, b))
    return params


def forward_cnn(params, x):
    activations = x
    for w, _ in params:
        outputs = conv(activations, w)  # + b
        activations = jax.nn.relu(outputs)
    return jax.nn.sigmoid(outputs)


def forward_mlp(params, x):
    activations = x
    for w, b in params:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    return jax.nn.sigmoid(outputs)


def init_params(config, rng):
    return {"cnn": init_cnn(config, rng), "mlp": init_mlp(config["layer_sizes"], rng)}


# model
def model(params, x):
    x = forward_cnn(params["cnn"], x)
    x = x.reshape(x.shape[0], -1)
    x = forward_mlp(params["mlp"], x)
    return x


def loss_fn(params, x, y):
    pred = model(params, x)  # batch_size x image_dim
    return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))


def predict(params, x):
    preds = model(params, x)
    return (preds > 0.5).astype(int)


strides = (3, 3)
conv = lambda x, w: jax.lax.conv_general_dilated(x, w, strides, padding="SAME")
