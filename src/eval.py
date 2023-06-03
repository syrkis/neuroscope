"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
from src.model import loss_fn, mse, bce, forward, soft_f1, focal_loss

# function for computing pearson's correlation coefficient for each voxel of a subject's fMRI data
def pearsons_corr(pred, target):
    """return pearson's correlation coefficient for each voxel"""
    corr = np.zeros(pred.shape[1])
    for v in tqdm(range(pred.shape[1])):
        corr[v] = np.corrcoef(pred[:, v], target[:, v])[0, 1]
    return corr


# evaluate during training
def evaluate(params, train_data, val_data, get_batch, config, steps=2):
    """evaluate function"""
    batch_size = config['batch_size']
    train_losses, train_lh_losses, train_rh_losses, train_cat_losses = [], [], [], []
    val_losses, val_lh_losses, val_rh_losses, val_cat_losses = [], [], [], []
    for _ in range(steps):
        train_batch = get_batch(train_data, batch_size)
        train_pred = forward(params, train_batch[0])
        val_batch = get_batch(val_data, batch_size)
        val_pred = forward(params, val_batch[0])

        train_loss = loss_fn(params, train_batch)
        val_loss = loss_fn(params, val_batch)

        train_lh_loss = mse(train_pred[0], train_batch[1])
        val_lh_loss = mse(val_pred[0], val_batch[1])

        train_rh_loss = mse(train_pred[1], train_batch[2])
        val_rh_loss = mse(val_pred[1], val_batch[2])

        train_cat_loss = focal_loss(train_pred[2], train_batch[3])
        val_cat_loss = focal_loss(val_pred[2], val_batch[3])

        train_losses.append(train_loss)
        train_lh_losses.append(train_lh_loss)
        train_rh_losses.append(train_rh_loss)
        train_cat_losses.append(train_cat_loss)

        val_losses.append(val_loss)
        val_lh_losses.append(val_lh_loss)
        val_rh_losses.append(val_rh_loss)
        val_cat_losses.append(val_cat_loss)
        # test if any loss is nan
        if np.isnan(train_loss) or np.isnan(val_loss) or np.isnan(train_lh_loss) or np.isnan(val_lh_loss) or np.isnan(train_rh_loss) or np.isnan(val_rh_loss) or np.isnan(train_cat_loss) or np.isnan(val_cat_loss):
            print('nan loss')
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

def save_best_model(params, val_loss, best_val_loss, subject, hem):
    """save best model"""
    if hem == 'lh' and val_loss < best_val_loss:
        jnp.save(f"results/models/{subject}_lh_best_model.npy", params)
    elif hem == 'rh' and val_loss < best_val_loss:
        jnp.save(f"results/models/{subject}_rh_best_model.npy", params)
    return best_val_loss