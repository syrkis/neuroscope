# model.py
#     neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
from jax import random, grad, jit, vmap


# init_params
def init_mlp(layer_sizes, rng):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = random.normal(rng, (n_in, n_out)) * jnp.sqrt(2 / n_in)
        b = random.normal(rng, (n_out,)) * jnp.sqrt(2 / n_in)
        params.append((w, b))
    return params


def init_cnn(kernel_sizes, rng):  # kernel dim is [out_channels, in_channels, kernel_size, kernel_size]
    params = []
    for c_in, c_out, k in kernel_sizes:
        w = random.normal(rng, (c_out, c_in, k, k)) * jnp.sqrt(2 / (k * k * c_in))
        b = random.normal(rng, (c_out,)) * jnp.sqrt(2 / (k * k * c_in))
        params.append((w, b))
    return params


def forward_cnn(params, x):
    activations = x
    for w, b in params:
        outputs = conv2d(activations, w) # + b
        activations = jax.nn.relu(outputs)
    return jax.nn.sigmoid(outputs)

def forward_mlp(params, x):
    activations = x
    for w, b in params:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    return jax.nn.sigmoid(outputs)


def init_params(layer_sizes, rng):
    params = {'mlp': init_mlp(layer_sizes['mlp'], rng), 'cnn': init_cnn(layer_sizes['cnn'], rng)}
    return params


# model
def model(params, x):
    x = forward_cnn(params['cnn'], x)
    x = x.reshape(x.shape[0], -1)
    x = forward_mlp(params['mlp'], x)
    return x



def loss_fn(params, x, y):
    pred = model(params, x)  # batch_size x image_dim
    return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))


def predict(params, x):  
    preds = model(params, x)
    return (preds > 0.5).astype(int)


stride = (1, 2, 2, 1)
conv = lambda x, w, d: jax.lax.conv_general_dilated(x, w, (1, 2, 2, 1), padding='SAME')
conv2d = lambda x, w: conv(x, w, 2)
conv3d = lambda x, w: conv(x, w, 3)