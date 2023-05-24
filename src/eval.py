"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import numpy as np
from src.model import train_loss_fn, infer_loss_fn


# evaluate
def evaluate(params, train_data, val_data, get_batch, config, steps=2):
    batch_size = config['batch_size']
    train_loss, val_loss = [], []
    for step in range(steps):
        train_batch = get_batch(train_data, batch_size)
        val_batch = get_batch(val_data, batch_size)
        train_loss.append(train_loss_fn(params, train_batch))
        val_loss.append(infer_loss_fn(params, val_batch))
    return {'train_loss': np.mean(train_loss), 'val_loss': np.mean(val_loss)}

def save_if_best(params, best_loss, val_loss):
    if val_loss < best_loss:
        best_loss = val_loss
        np.save('best_params.npy', params)
    return best_loss