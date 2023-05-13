# data.py
#     algonauts project
# by: Noah Syrkis

# imports
from src.utils import get_files, DATA_DIR, cat_id_to_name, coco_cat_id_to_vec_index
from src.fmri import roi_response_to_image, lh_fmri, rh_fmri, get_roi_mask
from jax import numpy as jnp
import pandas as pd
import numpy as np
from PIL import Image
import ast
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# batch_loader
def get_loaders(args, config):  # TODO: allow for loading and mixing multiple subjects
    subject, n_samples, batch_size, = args.subject, args.n_samples, args.batch_size
    image_size = config['data']['image_size']
    _, _, img_files = get_files(args.subject)
    train_idxs, test_idxs = train_test_split(range(len(img_files)), test_size=0.2, random_state=42)
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2, random_state=42)
    meta_data = get_meta_data()
    train_loader = get_loader(subject, batch_size, meta_data, n_samples, train_idxs, image_size, args)
    val_loader = get_loader(subject, batch_size, meta_data, n_samples, val_idxs, image_size, args)
    test_loader = get_loader(subject, batch_size, meta_data, n_samples, test_idxs, image_size, args)
    return train_loader, val_loader, test_loader


# get batch for subject. TODO: make subject mixed batches. fmri dimensions might be subject specific.
def get_loader(subject, batch_size, meta_data, n_samples, idxs, image_size, args):
    lh_fmri_roi = lh_fmri[:, get_roi_mask(args.roi, 'left')]
    rh_fmri_roi = rh_fmri[:, get_roi_mask(args.roi, 'right')]
    _, _, image_files = get_files(subject)
    n_samples = n_samples if n_samples else len(image_files)
    image_files = [image_files[i] for i in idxs]
    if n_samples < len(image_files):
        image_files = random.sample(image_files, n_samples)
    images = []
    # look up categories for each image
    coco_ids = [int(f.split('.')[0].split('-')[-1]) for f in image_files]
    categories = meta_data.iloc[coco_ids]['categories'].values
    supers = meta_data.iloc[coco_ids]['supercategory'].values
    captions = meta_data.iloc[coco_ids]['captions'].values
    for image_file in tqdm(image_files):
        images.append(np.array(preprocess(Image.open(image_file), image_size)))
    images = jnp.array(images)
    while True:
        perm = np.random.permutation(len(image_files))
        for i in range(0, len(image_files), batch_size):
            idxs = perm[i:i + batch_size]
            # sample random category from each category list
            cat = jnp.array([c_to_one_hot(c) for c in categories[idxs]])
            # reshape images to (batch_size, 3, image_size, image_size)  bec
            img = images[idxs].reshape((len(idxs), 3, image_size, image_size))  # don't hard code
            yield img, cat, supers[idxs], captions[idxs], lh_fmri_roi[idxs], rh_fmri_roi[idxs]


def preprocess(image, image_size):
    image = image.resize((image_size, image_size))  # TODO: resize images perhaps
    image = np.array(image)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def c_to_one_hot(categories):  # there are 80 categories, but the ids range from 1-90 (skipping 10) TODO: deal with this perhaps
    one_hot = np.zeros(len(cat_id_to_name))
    for cat in categories:
        one_hot[coco_cat_id_to_vec_index[cat]] = 1
    return jnp.array(one_hot)


def file_name_has_valid_coco_id(file_name, coco_ids):
    id_from_file = file_name.split('/')[-1].split('.')[0].split('_')[-1].split('-')[-1]
    return int(id_from_file) in coco_ids


def make_meta_data():  # if you want to make the meta data csv from scratch
    coco_df = pd.read_csv(os.path.join(DATA_DIR, 'coco_meta_data.csv'), index_col='cocoId')
    coco_df['categories'] = coco_df['categories'].apply(lambda x: ast.literal_eval(x))
    coco_df['supercategory'] = coco_df['supercategory'].apply(lambda x: ast.literal_eval(x))
    coco_df['captions'] = coco_df['captions'].apply(lambda x: ast.literal_eval(x))
    nsd_df = pd.read_csv(os.path.join(DATA_DIR, 'nsd_stim_info_merged.csv'))
    meta_data = coco_df.loc[nsd_df['cocoId']]
    meta_data['nsdId'] = nsd_df.index
    meta_data['split'] = nsd_df['cocoSplit']
    meta_data.to_csv(os.path.join(DATA_DIR, 'meta_data.csv'))

def get_meta_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'meta_data.csv'), index_col='nsdId')
    df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x))
    df['supercategory'] = df['supercategory'].apply(lambda x: ast.literal_eval(x))
    df['captions'] = df['captions'].apply(lambda x: ast.literal_eval(x))
    return df


def get_split_idxs(split, meta_data):
    if split == 'train':
        split = 'train2017'
    elif split == 'valid':
        split = 'val2017'
    mask = meta_data['split'] == split
    return meta_data[mask].index.values  # might be wrong