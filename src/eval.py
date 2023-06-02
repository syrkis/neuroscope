"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import numpy as np
from src.model import loss_fn, mse, bce, forward


# evaluate
def evaluate(params, train_data, val_data, get_batch, config, steps=2):
    batch_size = config['batch_size']
    train_loss, train_lh_loss, train_rh_loss, train_cat_loss = [], [], [], []
    val_loss, val_lh_loss, val_rh_loss, val_cat_loss = [], [], [], []
    for _ in range(steps):
        train_batch = get_batch(train_data, batch_size)
        train_pred = forward(params, train_batch[0])
        val_batch = get_batch(val_data, batch_size)
        val_pred = forward(params, val_batch[0])
        train_loss.append(loss_fn(params, train_batch))
        val_loss.append(loss_fn(params, val_batch))
        train_lh_loss.append(mse(train_pred[0], train_batch[1]))
        val_lh_loss.append(mse(val_pred[0], val_batch[1]))
        train_rh_loss.append(mse(train_pred[1], train_batch[2]))
        val_rh_loss.append(mse(val_pred[1], val_batch[2]))
        train_cat_loss.append(bce(train_pred[2], train_batch[3]))
        val_cat_loss.append(bce(val_pred[2], val_batch[3]))
    return {
        'train_loss': np.mean(train_loss),
        'val_loss': np.mean(val_loss),
        'train_lh_loss': np.mean(train_lh_loss),
        'val_lh_loss': np.mean(val_lh_loss),
        'train_rh_loss': np.mean(train_rh_loss),
        'val_rh_loss': np.mean(val_rh_loss),
        'train_cat_loss': np.mean(train_cat_loss),
        'val_cat_loss': np.mean(val_cat_loss),
    }
    