# data.py
#     algonauts project
# by: Noah Syrkis

# imports
from jax import numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.utils import get_files
from src.fmri import lh_fmri, rh_fmri, get_multi_roi_mask
from src.coco import preprocess, get_meta_data, c_to_one_hot


# batch_loader
def get_loaders(args, config): 
    """return a test data loader, and a k-fold cross validation generator"""
    meta_data = get_meta_data()
    img_files = [f for f in get_files(args.subject) if f.endswith(".png")][: config['n_samples']]
    images = jnp.array([preprocess(Image.open(f), config['image_size']) for f in tqdm(img_files)])
    train_idxs, test_idxs = map(jnp.array, train_test_split(range(len(images)), test_size=0.2, random_state=42))
    train_img_files = [img_files[idx] for idx in train_idxs]
    folds = get_folds(images[train_idxs], args, meta_data, train_img_files, k=config['k_folds'])
    test_img_files = [img_files[idx] for idx in test_idxs]
    test_data = get_data(images[test_idxs], args, meta_data, test_img_files)
    return folds, test_data


# cross validation
# TODO: split into folds, and iterate through and concatenate.
def get_folds(images, args, meta_data, img_files, k=5):
    """return a k-fold cross validation generator"""
    folds = []
    n_samples = len(images) // k * k
    fold_idxs = np.array_split(np.random.permutation(n_samples), k)
    for i in range(k):
        fold = get_data(images[fold_idxs[i]], args, meta_data, [img_files[idx] for idx in fold_idxs[i]])
        folds.append(fold)
    return folds


def get_data(images, args, meta_data, img_files):
    """return a data loader combining images and fmri data, and adding COCO stuff"""
    lh_fmri_roi = lh_fmri[:, get_multi_roi_mask(args.rois, "left")]
    rh_fmri_roi = rh_fmri[:, get_multi_roi_mask(args.rois, "right")]
    fmri = jnp.concatenate((lh_fmri_roi, rh_fmri_roi), axis=1)

    coco_ids = [int(f.split(".")[0].split("-")[-1]) for f in img_files]  # coco meta ids
    cats = meta_data.iloc[coco_ids]["categories"].values  # category info
    cats = jnp.array([c_to_one_hot(c) for c in cats])  # one-hot encoding
    perm = np.random.permutation(len(img_files))  # randomize order of images
    return images[perm], cats[perm], fmri[perm]  # supers[perm], captions[perm], fmri[perm]
    # supers = meta_data.iloc[coco_ids]["supercategory"].values  # supercategory info
    # captions = meta_data.iloc[coco_ids]["captions"].values  # caption info