"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import numpy as np
from src.model import train_loss, infer_loss


# evaluate
def evaluate(params, train_data, val_data, get_batch, config, steps=2):
    batch_size = config['batch_size']
    train_loss = [train_loss(params, get_batch(train_data, batch_size)) for _ in range(steps)]
    val_loss = [infer_loss(params, get_batch(val_data, batch_size)) for _ in range(steps)]
    return {'train_loss': np.mean(train_loss), 'val_loss': np.mean(val_loss)}

def save_if_best(params, best_loss, val_loss):
    if val_loss < best_loss:
        best_loss = val_loss
        np.save('best_params.npy', params)
    return best_loss