# data.py
#     neuroscope project
# by: Noah Syrkis

# imports
import os
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split, KFold
import jax.numpy as jnp
import pickle
from src.utils import DATA_DIR, get_metadata_sources


# functions
def load_subjects(subjects: list, image_size: int=32, precision=jnp.float32) -> dict:
    return {subject: load_subject(subject, image_size, precision) for subject in subjects}
    

def load_subject(subject: str, image_size: int=32, precision=jnp.float32) -> tuple:
    path = os.path.join(DATA_DIR, 'algonauts', subject, 'training_split')
    n_samples = len([f for f in os.listdir(os.path.join(path, 'training_images')) if f.endswith('.png')])
    train_idx, _ = train_test_split(np.arange(n_samples), test_size=0.2, random_state=42)
    return load_split(path, train_idx, image_size, subject, precision=precision)


def load_split(path: str, split_idx: int, image_size: int, subject: str, precision) -> tuple:
    lh_fmri = np.load(os.path.join(path, 'training_fmri', 'lh_training_fmri.npy'))[split_idx]
    rh_fmri = np.load(os.path.join(path, 'training_fmri', 'rh_training_fmri.npy'))[split_idx]
    images, metadata = load_coco(path, split_idx, image_size, subject)
    return lh_fmri.astype(precision), rh_fmri.astype(precision), images.astype(precision), metadata


def load_metadata(image_files: list, subject: str) -> list:
    metadata_cache = os.path.join(DATA_DIR, 'cache', f'metadata_{subject}.pkl')
    if os.path.exists(metadata_cache):
        with open(metadata_cache, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata_sources = get_metadata_sources()
        metadata = [get_metadata(image_file, metadata_sources) for image_file in image_files]
        with open(metadata_cache, 'wb') as f:
            pickle.dump(metadata, f)
    return metadata


def load_images(image_files: list, path: str, image_size: int, subject: str) -> np.ndarray:
    image_cache = os.path.join(DATA_DIR, 'cache', f'images_{subject}_{image_size}.npy')
    if os.path.exists(image_cache):
        images = np.load(image_cache)
    else:
        load_image = lambda f: cv2.resize(cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (image_size, image_size)) / 255.
        image_paths = [os.path.join(path, 'training_images', f) for f in image_files]
        images = np.array([load_image(image_path) for image_path in tqdm(image_paths)])
        np.save(image_cache, images)
    return images


def load_coco(path: str, split_idx: int, image_size: int, subject: str) -> tuple:
    image_files = sorted([f for f in os.listdir(os.path.join(path, 'training_images')) if f.endswith('.png')])
    image_files = [image_files[i] for i in split_idx]
    metadata = load_metadata(image_files, subject)
    images = load_images(image_files, path, image_size, subject)
    return images, metadata


def get_metadata(image_file: str, metadata_sources: tuple) -> tuple:
    coco_instances, coco_captions, nsd_stim_info = metadata_sources
    nsd_id = int(image_file.split('_')[1].split('.')[0].split('-')[1])
    coco_id = nsd_stim_info.loc[nsd_stim_info['nsdId'] == nsd_id, 'cocoId'].iloc[0]
    annotation_ids = coco_instances.getAnnIds(imgIds=coco_id)
    annotations = coco_instances.loadAnns(annotation_ids)
    category_ids = [annotation['category_id'] for annotation in annotations]
    categories = [coco_instances.loadCats(category_id)[0]['name'] for category_id in category_ids]
    caption_ids = coco_captions.getAnnIds(imgIds=coco_id)
    captions = [annotation['caption'] for annotation in coco_captions.loadAnns(caption_ids)]
    return coco_id, categories, captions


def make_batches(lh_fmri: list, rh_fmri: list, images: list , subject_idx, batch_size: int, n_subjects: int) -> tuple:
    lh_fmri = np.array(lh_fmri)
    rh_fmri = np.array(rh_fmri)
    images = np.array(images)
    while True:
        perm = np.random.permutation(len(images) // batch_size * batch_size)
        for i in range(0, len(perm), batch_size):
            batch_perm = perm[i:i + batch_size]
            lh_batch = lh_fmri[batch_perm]
            #expanded_lh = expand(lh_batch, subject_idx[batch_perm], n_subjects)
            rh_batch = rh_fmri[batch_perm]
            #expanded_rh = expand(rh_batch, subject_idx[batch_perm], n_subjects)
            image_batch = images[batch_perm]
            yield lh_batch, rh_batch, image_batch, subject_idx[batch_perm]

""" def expand(A, v, k):
    N, M = A.shape
    B = jnp.zeros((N, M, k))
    mask = jnp.arange(k) == v[:, None]
    B = jnp.where(mask[:, None, :], A[:, :, None], B)
    return B """

def combine_subjects(subjects, cfg):
    # subjects is a dict of (lh_fmri, rh_fmri, images) tuples
    lh_fmri = np.concatenate([subject[0] for subject in subjects.values()])
    rh_fmri = np.concatenate([subject[1] for subject in subjects.values()])
    images = np.concatenate([subject[2] for subject in subjects.values()])
    subject_idx = np.concatenate([np.ones(len(subject[0])) * i for i, subject in enumerate(subjects.values())])
    return lh_fmri, rh_fmri, images, subject_idx


def make_kfolds(subjects_data, cfg, n_splits=5):
    # subject data is a dict of (lh_fmri, rh_fmri, images) tuples
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    combined_subjects = combine_subjects(subjects_data, cfg)
    for fold in kf.split(combined_subjects[0]):
        train_idx, val_idx = fold
        train_lh, train_rh = combined_subjects[0][train_idx], combined_subjects[1][train_idx]
        train_images = combined_subjects[2][train_idx]
        train_subject_idx = combined_subjects[3][train_idx]
        val_lh, val_rh = combined_subjects[0][val_idx], combined_subjects[1][val_idx]
        val_images = combined_subjects[2][val_idx]
        val_subject_idx = combined_subjects[3][val_idx]
        train_batches = make_batches(train_lh, train_rh, train_images, train_subject_idx, cfg['batch_size'], len(subjects_data))
        val_batches = make_batches(val_lh, val_rh, val_images, val_subject_idx, cfg['batch_size'], len(subjects_data))
        yield train_batches, val_batches