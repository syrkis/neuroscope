"""evaluates model performance"""
# eval.py
#   evaluates model performance
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax import vmap
from tqdm import tqdm
import haiku as hk
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from functools import partial
from src.model import mse, bce, soft_f1, focal_loss, network_fn


def pearsonr(x, y):
    corr = jnp.corrcoef(x, y)
    return corr[0, 1]

# function for computing pearson's correlation coefficient for each voxel of a subject's fMRI data
def corr(pred, target):
    hem_corr = vmap(pearsonr)(pred.T, target.T)
    return hem_corr


# evaluate during training
def evaluate(params, rng, train_data, val_data, get_batch, config, steps=5):
    """evaluate function"""
    forward = hk.transform(partial(network_fn, config=config))
    train_lh_losses, train_rh_losses, train_cat_losses = [], [], []
    train_lh_corrs, train_rh_corrs = [], []
    val_lh_losses, val_rh_losses, val_cat_losses = [], [], []
    val_lh_corrs, val_rh_corrs = [], []
    for _ in range(steps):
        train_batch = get_batch(train_data, config.batch_size)
        train_pred = forward.apply(params, rng, train_batch[0])

        train_lh_loss = mse(train_pred[0], train_batch[1])
        train_rh_loss = mse(train_pred[1], train_batch[2])
        train_cat_loss = focal_loss(train_pred[2], train_batch[3])
        train_lh_corr = corr(train_pred[0], train_batch[1])
        train_rh_corr = corr(train_pred[1], train_batch[2])

        train_lh_losses.append(train_lh_loss)
        train_rh_losses.append(train_rh_loss)
        train_cat_losses.append(train_cat_loss)
        train_lh_corrs.append(jnp.median(train_lh_corr))
        train_rh_corrs.append(jnp.median(train_rh_corr))

        val_batch = get_batch(val_data, config.batch_size)
        val_pred = forward.apply(params, rng, val_batch[0], training=False)

        val_lh_loss = mse(val_pred[0], val_batch[1])
        val_rh_loss = mse(val_pred[1], val_batch[2])
        val_cat_loss = focal_loss(val_pred[2], val_batch[3])
        val_lh_corr = corr(val_pred[0], val_batch[1])
        val_rh_corr = corr(val_pred[1], val_batch[2])

        val_lh_losses.append(val_lh_loss)
        val_rh_losses.append(val_rh_loss)
        val_cat_losses.append(val_cat_loss)
        val_lh_corrs.append(jnp.median(val_lh_corr))   # MEDIAN correlation across voxels
        val_rh_corrs.append(jnp.median(val_rh_corr))   # MEDIAN correlation across voxels

        # test if any loss is nan
        if np.isnan(train_lh_loss) or np.isnan(val_lh_loss) or np.isnan(train_rh_loss) or np.isnan(val_rh_loss) or np.isnan(train_cat_loss) or np.isnan(val_cat_loss):
            print('nan loss')
    return {
        'train_lh_loss': np.mean(train_lh_losses),
        'val_lh_loss': np.mean(val_lh_losses),
        'train_rh_loss': np.mean(train_rh_losses),
        'val_rh_loss': np.mean(val_rh_losses),
        'train_cat_loss': np.mean(train_cat_losses),
        'val_cat_loss': np.mean(val_cat_losses),
        'train_lh_corr': np.mean(train_lh_corrs),
        'val_lh_corr': np.mean(val_lh_corrs),
        'train_rh_corr': np.mean(train_rh_corrs),
        'val_rh_corr': np.mean(val_rh_corrs),
        #'algonauts_lh_baseline_corr': algonauts_baseline[config['subject']]['lh'],
        #'algonauts_rh_baseline_corr': algonauts_baseline[config['subject']]['rh'],
        #'algonauts_train_lh_corr': np.mean(algonauts_train_lh_corr),
        #'algonauts_val_lh_corr': np.mean(algonauts_val_lh_corr),
        #'algonauts_train_rh_corr': np.mean(algonauts_train_rh_corr),
        #'algonauts_val_rh_corr': np.mean(algonauts_val_rh_corr),
    }

def save_best_model(params, val_loss, best_val_loss, subject, hem):
    """save best model"""
    if hem == 'lh' and val_loss < best_val_loss:
        jnp.save(f"models/{subject}_lh_best_model.npy", params)
    elif hem == 'rh' and val_loss < best_val_loss:
        jnp.save(f"models/{subject}_rh_best_model.npy", params)
    return best_val_loss


def get_algonauts_model(subject):
    """get algonauts model"""
    lh_model = pickle.load(open(f"models/{subject}_lh_algonauts_model.pkl", 'rb'))
    rh_model = pickle.load(open(f"models/{subject}_rh_algonauts_model.pkl", 'rb'))
    return {'lh': lh_model, 'rh': rh_model}

def algonauts_baseline(fold):
    train_data, val_data = fold
    img, lh, rh, _ = train_data
    lh_model = LinearRegression().fit(img, lh)
    rh_model = LinearRegression().fit(img, rh)
    train_lh_pred = lh_model.predict(img)
    train_rh_pred = rh_model.predict(img)
    train_lh_corr = corr(train_lh_pred, lh)
    train_rh_corr = corr(train_rh_pred, rh)
    train_lh_corr = jnp.median(train_lh_corr)
    train_rh_corr = jnp.median(train_rh_corr)
    img, lh, rh, _ = val_data
    val_lh_pred = lh_model.predict(img)
    val_rh_pred = rh_model.predict(img)
    val_lh_corr = corr(val_lh_pred, lh)
    val_rh_corr = corr(val_rh_pred, rh)
    val_lh_corr = jnp.median(val_lh_corr)
    val_rh_corr = jnp.median(val_rh_corr)
    return {'linear_lh_train_corr': train_lh_corr, 'linear_rh_train_corr': train_rh_corr, 'linear_lh_val_corr': val_lh_corr, 'linear_rh_val_corr': val_rh_corr}

