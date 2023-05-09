# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
from src.model import init_params, model, loss_fn, predict
import jax
from jax import numpy as jnp
from jax import random, grad, jit, vmap
from tqdm import tqdm
import optax


def train(config, train_loader, val_loader):
    params = init_params(config['layer_sizes'], jax.random.PRNGKey(0))
    opt = optax.adam(config['lr'])
    opt_state = opt.init(params)
    params, metrics = train_steps(params, train_loader, val_loader, opt, opt_state, config['n_steps'])
    return params, metrics
    

def train_steps(params, train_loader, val_loader, opt, opt_state, n_steps):
    metrics = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [] }
    pbar = tqdm(range(n_steps))
    for step in pbar:
        x, y, _, _ = next(train_loader)
        x = x.mean(-1).reshape(x.shape[0], -1)   # TODO: this is a hack cos I'm flattening, but I don't wanna fltten
        params, opt_state = train_step(params, x, y, opt, opt_state)
        if step % (n_steps // 10) == 0:
            metrics = evaluate(params, train_loader, val_loader, metrics)
            pbar.set_description(f"train loss: {metrics['train_loss'][-1]:.4f}, train acc: {metrics['train_acc'][-1]:.4f}, val loss: {metrics['val_loss'][-1]:.4f}, val acc: {metrics['val_acc'][-1]:.4f}")
    return params, metrics


def train_step(params, x, y, opt, opt_state):
    grads = grad(loss_fn)(params, x, y)
    updates, opt_state = update(grads, opt, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


@jit
def update(grads, opt, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    return updates, opt_state


def evaluate(params, train_loader, valid_loader, metrics, steps=5):
    # metrics is dict of lists of val loss val acc train loss train acc
    train_loss, train_acc, valid_loss, valid_acc = [], [] ,[], []
    for _ in range(steps):
        train_x, train_y, _, _ = next(train_loader)
        train_x = train_x.mean(3).reshape(train_x.shape[0], -1)   # TODO: this is a hack cos I'm flattening, but I don't wanna fltten
        train_loss.append(loss_fn(params, train_x, train_y))
        train_acc.append(accuracy(params, train_x, train_y))
        valid_x, valid_y, _, _ = next(valid_loader)
        valid_x = valid_x.mean(3).reshape(valid_x.shape[0], -1)   # TODO: this is a hack cos I'm flattening, but I don't wanna fltten
        valid_loss.append(loss_fn(params, valid_x, valid_y))
        valid_acc.append(accuracy(params, valid_x, valid_y))
    for metric in metrics.keys():
        metrics[metric].append(jnp.mean(metric))
    return metrics


def accuracy(params, x, y):
    preds = predict(params, x)
    return jnp.mean(preds == y)
