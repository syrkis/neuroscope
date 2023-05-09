# train.py
#     neuroscope project
# by: Noah Syrkis

# imports
from src.model import init_params, model, loss_fn, predict
import jax
from jax import numpy as jnp
import numpy as np
from jax import random, grad, jit, vmap
from tqdm import tqdm
import optax


opt = optax.adam(1e-3)


def train(params, metrics, config, args, train_loader, val_loader):
    opt_state = opt.init(params)
    params, metrics = train_steps(params, metrics, train_loader, val_loader, opt_state, args.n_steps)
    return params, metrics
    

def train_steps(params, metrics, train_loader, val_loader, opt_state, n_steps):
    pbar = tqdm(range(n_steps))
    for step in pbar:
        x, y, _, _ = next(train_loader)
        params, opt_state = update(params, x, y, opt_state)
        if step % (n_steps // 100) == 0:
            metrics = evaluate(params, train_loader, val_loader, metrics)
            pbar.set_description(f"train loss: {metrics['train_loss'][-1]:.4f}, train acc: {metrics['train_acc'][-1]:.4f}, val loss: {metrics['val_loss'][-1]:.4f}, val acc: {metrics['val_acc'][-1]:.4f}")
    return params, metrics


@jit
def update(params, x, y, opt_state):
    grads = grad(loss_fn)(params, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


def accuracy(params, x, y):
    preds = predict(params, x)
    return jnp.mean(preds == y)


def evaluate(params, train_loader, valid_loader, metrics, steps=5):
    # metrics is dict of lists of val loss val acc train loss train acc
    train_loss, train_acc, valid_loss, valid_acc = [], [] ,[], []
    for _ in range(steps):
        train_x, train_y, _, _ = next(train_loader)
        train_loss.append(loss_fn(params, train_x, train_y))
        train_acc.append(accuracy(params, train_x, train_y))
        valid_x, valid_y, _, _ = next(valid_loader)
        valid_loss.append(loss_fn(params, valid_x, valid_y))
        valid_acc.append(accuracy(params, valid_x, valid_y))
    metrics['train_loss'].append(np.mean(train_loss))
    metrics['train_acc'].append(np.mean(train_acc))
    metrics['val_loss'].append(np.mean(valid_loss))
    metrics['val_acc'].append(np.mean(valid_acc))
    return metrics

