"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import jax.numpy as jnp
import numpy as np
from src.model import loss_fn


# evaluate
def evaluate(params, train_loader, val_loader, steps=20):
    train_loss, val_loss = [], []
    for _ in range(steps):
        img, cat, sup, cap, fmri = next(train_loader)  # training
        train_loss.append(loss_fn(params, img, cat, fmri))
        img, cat, sup, cap, fmri = next(val_loader)  # validation
        val_loss.append(loss_fn(params, img, cat, fmri))
    return dict(
        train_loss=np.mean(train_loss),
        val_loss=np.mean(val_loss),
    )