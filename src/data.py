# data.py
#     algonauts project
# by: Noah Syrkis

# imports
from src.utils import get_files
from src.fmri import lh_fmri, rh_fmri, get_multi_roi_mask
from src.coco import preprocess, get_meta_data, c_to_one_hot
from jax import numpy as jnp
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# batch_loader
def get_loaders(args, config):  # TODO: allow for loading and mixing multiple subjects
    image_size = config["data"]["image_size"]
    _, _, img_files = get_files(args.subject)
    train_idxs, test_idxs = train_test_split(
        range(len(img_files)), test_size=0.2, random_state=42
    )
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2, random_state=42)
    meta_data = get_meta_data()
    train_loader = get_loader(
        args.subject,
        args.batch_size,
        meta_data,
        args.n_samples,
        train_idxs,
        image_size,
        args,
    )
    val_loader = get_loader(
        args.subject,
        args.batch_size,
        meta_data,
        args.n_samples,
        val_idxs,
        image_size,
        args,
    )
    test_loader = get_loader(
        args.subject,
        args.batch_size,
        meta_data,
        args.n_samples,
        test_idxs,
        image_size,
        args,
    )
    return train_loader, val_loader, test_loader


# get batch for subject. TODO: make subject mixed batches. fmri dimensions might be subject specific.
def get_loader(subject, batch_size, meta_data, n_samples, idxs, image_size, args):
    lh_fmri_roi = lh_fmri[:, get_multi_roi_mask(args.rois, "left")]
    rh_fmri_roi = rh_fmri[:, get_multi_roi_mask(args.rois, "right")]
    fmri = np.concatenate((lh_fmri_roi, rh_fmri_roi), axis=1)
    _, _, image_files = get_files(subject)
    n_samples = n_samples if n_samples else len(image_files)
    image_files = [image_files[i] for i in idxs]
    if n_samples < len(image_files):
        image_files = random.sample(image_files, n_samples)
    image_files = image_files[: len(image_files) // batch_size * batch_size]
    images = []
    # look up categories for each image
    coco_ids = [int(f.split(".")[0].split("-")[-1]) for f in image_files]
    categories = meta_data.iloc[coco_ids]["categories"].values
    supers = meta_data.iloc[coco_ids]["supercategory"].values
    captions = meta_data.iloc[coco_ids]["captions"].values
    for image_file in tqdm(image_files):
        images.append(np.array(preprocess(Image.open(image_file), image_size)))
    images = jnp.array(images)
    while True:
        perm = np.random.permutation(len(image_files))
        for i in range(0, len(image_files), batch_size):
            idxs = perm[i : i + batch_size]
            # sample random category from each category list
            cat = jnp.array([c_to_one_hot(c) for c in categories[idxs]])
            # reshape images to (batch_size, 3, image_size, image_size)  bec
            img = images[idxs].reshape(
                (len(idxs), 3, image_size, image_size)
            )  # don't hard code
            fmri = np.concatenate((lh_fmri_roi, rh_fmri_roi), axis=1)
            yield img, cat, supers[idxs], captions[idxs], fmri[idxs]


def get_split_idxs(split, meta_data):
    if split == "train":
        split = "train2017"
    elif split == "valid":
        split = "val2017"
    mask = meta_data["split"] == split
    return meta_data[mask].index.values  # might be wrong
