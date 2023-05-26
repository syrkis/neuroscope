# data.py
#     neuroscope project
# by: Noah Syrkis

# imports
from jax import numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utils import get_files
from src.fmri import get_multi_roi_mask, get_fmri  # NOTE: This is the only import from fmri.py
from src.coco import preprocess, get_meta_data, c_to_one_hot


# batch_loader
def get_data(args, config): 
    """return dictionary of data loaders for each subject"""
    data = {subject: None for subject in args.subjects.split(",")}
    meta_data = get_meta_data()
    for subject in args.subjects.split(","):
        """return a test data loader, and a k-fold cross validation generator"""
        img_files = [f for f in get_files(subject) if f.endswith(".png")][: config['n_samples']]
        images = jnp.array([preprocess(Image.open(f), config['image_size']) for f in tqdm(img_files)])

        train_idxs, test_idxs = map(jnp.array, train_test_split(range(len(images)), test_size=0.1, random_state=42))
        train_img_files = [img_files[idx] for idx in train_idxs.tolist()]
        folds = get_folds(images[train_idxs], args, meta_data, train_img_files, subject, k=config['k_folds'])

        test_img_files = [img_files[idx] for idx in test_idxs.tolist()]
        test_data = get_subject_data(images[test_idxs], args, meta_data, test_img_files, subject)

        data[subject] = (folds, test_data)
    return data


# TODO: make function that returns mixed data
def get_mixed_subject_data(args, config):
    """return a data loader combining images and fmri data, and adding COCO stuff"""
    data = get_data(args, config)
    mixed_data = []
    for key, value in data.items():
        mixed_data.append(value)


# cross validation
def get_folds(images, args, meta_data, img_files, subject, k=5):
    """return a k-fold cross validation generator"""
    folds = []
    # ensure that each fold has the same number of samples
    n_samples = (len(images) // k) * k
    fold_idxs = np.array_split(np.random.permutation(n_samples), k)
    for i in range(k):
        fold_img_files = [img_files[idx] for idx in fold_idxs[i]]
        fold = get_subject_data(images[fold_idxs[i]], args, meta_data, fold_img_files, subject)
        folds.append(fold)
    return folds


def get_subject_data(images, args, meta_data, img_files, subject):
    """return a data loader combining images and fmri data, and adding COCO stuff"""
    lh_fmri, rh_fmri = get_fmri(subject)
    lh = lh_fmri[:, get_multi_roi_mask(subject, args.rois, "left")]
    rh = rh_fmri[:, get_multi_roi_mask(subject, args.rois, "right")]

    coco_ids = [int(f.split(".")[0].split("-")[-1]) for f in img_files]  # coco meta ids
    cats = meta_data.iloc[coco_ids]["categories"].values  # category info
    cats = jnp.array([c_to_one_hot(c) for c in cats])  # one-hot encoding
    perm = np.random.permutation(len(img_files))  # randomize order of images
    return images[perm], cats[perm], lh[perm], rh[perm]  # supers[perm], captions[perm], fmri[perm]
    # supers = meta_data.iloc[coco_ids]["supercategory"].values  # supercategory info
    # captions = meta_data.iloc[coco_ids]["captions"].values  # caption info