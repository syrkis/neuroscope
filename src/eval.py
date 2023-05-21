"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import jax.numpy as jnp
import numpy as np
from src.model import loss_fn


# evaluate
def evaluate(params, train_data, val_data, get_batch_fn, steps=2):
    train_loss, val_loss = [], []
    for _ in range(steps):
        img, cat, fmri = get_batch_fn(train_data, 1)
        train_loss.append(loss_fn(params, img, cat, fmri))
        img, cat, fmri = get_batch_fn(val_data, 1)
        val_loss.append(loss_fn(params, img, cat, fmri))
    return dict(
        train_loss=np.mean(train_loss),
        val_loss=np.mean(val_loss),
    )

def save_if_best(params, best_loss, val_loss):
    if val_loss < best_loss:
        best_loss = val_loss
        np.save('best_params.npy', params)
    return best_loss