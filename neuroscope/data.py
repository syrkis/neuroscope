# data.py
#     neuroscope project
# by: Noah Syrkis

# imports
import os

import cv2
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm
from cachier import cachier

import neuroscope.utils as utils

# %% Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "data")
COCO_DIR = os.path.join(DATA_DIR, "coco", "annotations")
INSTANCES_DIR = os.path.join(COCO_DIR, "instances_train2017.json")
CAPTIONS_DIR = os.path.join(COCO_DIR, "captions_train2017.json")
ALGONAUTS_DIR = os.path.join(DATA_DIR, "algonauts")


def subject_fn(cfg):
    lh = hemisphere_fn(cfg.subj, "lh")
    rh = hemisphere_fn(cfg.subj, "rh")
    maps = maps_fn(cfg.subj)
    coco = coco_fn(cfg)
    return utils.Subject(lh=lh, rh=rh, maps=maps, coco=coco)


def bold_fn(subj, hem):
    file = os.path.join(
        ALGONAUTS_DIR,
        subj,
        f"training_split/training_fmri/{hem}_training_fmri.npy",
    )
    bold = jnp.array(np.load(file))
    bold = (bold - bold.mean()) / bold.std()
    return bold


def maps_fn(subj):
    mapping_files = [
        os.path.join(ALGONAUTS_DIR, subj, "roi_masks", f)
        for f in os.listdir(os.path.join(ALGONAUTS_DIR, subj, "roi_masks"))
        if "mapping" in f  # and "streams" not in f
    ]

    maps = {
        k.split("/")[-1].split(".")[0].split("_")[1]: v
        for k, v in zip(
            mapping_files,
            list(map(lambda x: np.load(x, allow_pickle=True).item(), mapping_files)),
        )
    }
    return maps


def mask_fn(subj, hem, atlas):
    mask_files = [
        os.path.join(ALGONAUTS_DIR, subj, "roi_masks", f)
        for f in os.listdir(os.path.join(ALGONAUTS_DIR, subj, "roi_masks"))
        if f.startswith(hem) and atlas in f  # and "stream" not in f
    ]
    mask = {
        k.split("/")[-1].split(".")[1].split("_")[0]: v
        for k, v in zip(
            mask_files,
            list(map(lambda x: jnp.array(np.load(x)), mask_files)),
        )
    }
    return mask


def hemisphere_fn(subj, hem):
    fsa = mask_fn(subj, hem, "fsaverage")
    cha = mask_fn(subj, hem, "challenge")
    mask = utils.Mask(fsa=fsa, cha=cha)
    bold = bold_fn(subj, hem)
    return utils.Hemisphere(mask=mask, bold=bold)


@cachier()
def meta_data_fn():
    coco_instances = (
        COCO(INSTANCES_DIR)
        if "COCO_INSTANCES" not in globals()
        else eval("COCO_INSTANCES")
    )
    coco_captions = (
        COCO(CAPTIONS_DIR)
        if "COCO_CAPTIONS" not in globals()
        else eval("COCO_CAPTIONS")
    )
    nsd_stim_info = pd.read_csv(
        os.path.join(DATA_DIR, "nsd", "nsd_stim_info_merged.csv")
    )
    return coco_instances, coco_captions, nsd_stim_info


@cachier()
def imgs_fn(image_files, image_size):
    return jnp.array(
        list(
            map(
                lambda f: cv2.resize(
                    cv2.imread(f, cv2.IMREAD_GRAYSCALE),
                    (image_size, image_size),
                )
                / 255.0,
                tqdm(image_files),
            )
        )
    )


def coco_fn(cfg):
    coco_inst, coco_capt, nsd_stim_info = meta_data_fn()
    image_files = [
        os.path.join(ALGONAUTS_DIR, cfg.subj, "training_split", "training_images", f)
        for f in os.listdir(
            os.path.join(ALGONAUTS_DIR, cfg.subj, "training_split", "training_images")
        )
        if f.endswith(".png")
    ]
    imgs = imgs_fn(image_files, cfg.image_size)
    nsd_ids = [
        int(f.split("/")[-1].split("_")[1].split(".")[0].split("-")[1])
        for f in image_files
    ]
    coco_ids = [
        nsd_stim_info.loc[nsd_stim_info["nsdId"] == nsd_id, "cocoId"].iloc[0]
        for nsd_id in nsd_ids
    ]
    annot_ids = [coco_inst.getAnnIds(imgIds=coco_id) for coco_id in coco_ids]
    anots = [coco_inst.loadAnns(annot_id) for annot_id in annot_ids]
    cats = np.zeros((len(image_files), 90))  # largest cat id is 90 (starting at 0
    for idx, anot in enumerate(anots):
        cats[idx] = (
            np.array(  # minus 1 to make id start at 0 (will cause bug in future)
                [np.eye(cats.shape[1])[np.array(a["category_id"] - 1)] for a in anot]
            ).any(0)
        )
    coco = utils.COCO(imgs=imgs, meta=jnp.array(cats))
    return coco
