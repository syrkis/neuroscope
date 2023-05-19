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
def get_loaders(args, config):  # TODO: allow for loading and mixing multiple subjects
    """return a test data loader, and a k-fold cross validation generator"""
    image_size = config["data"]["image_size"]
    meta_data = get_meta_data()
    images = []
    img_files = [f for f in get_files(args.subject) if f.endswith(".png")][
        : args.n_samples
    ]
    for f in tqdm(img_files):
        images.append(preprocess(Image.open(f), image_size))
    images = jnp.array(images)
    train_idxs, test_idxs = map(
        jnp.array, train_test_split(range(len(images)), test_size=0.2, random_state=42)
    )
    test_loader = get_loader(
        images[test_idxs], args, meta_data, [img_files[idx] for idx in test_idxs]
    )
    return (
        k_fold_fn(
            images[train_idxs], args, meta_data, [img_files[idx] for idx in train_idxs]
        ),
        test_loader,
    )


# cross validation
def k_fold_fn(images, args, meta_data, img_files, k=5):
    """return a k-fold cross validation generator"""
    for i in range(k):
        train_idxs, val_idxs = map(
            jnp.array,
            train_test_split(range(len(images)), test_size=0.2, random_state=i),
        )
        train_loader = get_loader(
            images[train_idxs], args, meta_data, [img_files[idx] for idx in train_idxs]
        )
        val_loader = get_loader(
            images[val_idxs], args, meta_data, [img_files[idx] for idx in val_idxs]
        )
        yield train_loader, val_loader


# get batch for subject. TODO: mix subjects?
def get_loader(images, args, meta_data, img_files):
    """return a data loader combining images and fmri data, and adding COCO stuff"""

    lh_fmri_roi = lh_fmri[:, get_multi_roi_mask(args.rois, "left")]
    rh_fmri_roi = rh_fmri[:, get_multi_roi_mask(args.rois, "right")]
    fmri = jnp.concatenate((lh_fmri_roi, rh_fmri_roi), axis=1)

    coco_ids = [
        int(f.split(".")[0].split("-")[-1]) for f in img_files
    ]  # for getting coco meta data
    cats = meta_data.iloc[coco_ids]["categories"].values  # category info
    cats = jnp.array([c_to_one_hot(c) for c in cats])  # one-hot encoding
    supers = meta_data.iloc[coco_ids]["supercategory"].values  # supercategory info
    captions = meta_data.iloc[coco_ids]["captions"].values  # caption info

    while True:
        perm = np.random.permutation(len(img_files))  # randomize order of images
        perm = perm[
            : len(perm) - (len(perm) % args.batch_size)
        ]  # drop last batch if it's not full
        for i in range(0, len(perm), args.batch_size):
            idxs = perm[i : i + args.batch_size]
            yield images[idxs], cats[idxs], supers[idxs], captions[idxs], fmri[idxs]
