"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import numpy as np
from src.model import train_loss, infer_loss


# evaluate
def evaluate(params, train_data, val_data, get_batch, config, steps=4):
    return dict(
        train_loss=train_loss(params, get_batch(train_data, config['batch_size'] * steps)),
        val_loss=infer_loss(params, get_batch(val_data, config['batch_size'] * steps))
    )

def save_if_best(params, best_loss, val_loss):
    if val_loss < best_loss:
        best_loss = val_loss
        np.save('best_params.npy', params)
    return best_loss
