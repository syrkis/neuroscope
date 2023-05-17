# coco.py
#   coco data set stuff
# by: Noah Syrkis

# imports
from src.utils import DATA_DIR, cat_id_to_name, coco_cat_id_to_vec_index
from jax import numpy as jnp
import os
import ast
import numpy as np
import pandas as pd

# functions for coco data sets, including loading and preprocessing, making meta data files, etc.
def preprocess(image, image_size):
    image = image.resize((image_size, image_size))
    image = np.array(image)
    image = image / 255.0
    image = image.astype(jnp.float32)
    return image


def c_to_one_hot(categories):
    # there are 80 categories, but the ids range from 1-90 (skipping 10) TODO: deal with this perhaps
    one_hot = np.zeros(len(cat_id_to_name))
    for cat in categories:
        one_hot[coco_cat_id_to_vec_index[cat]] = 1
    return jnp.array(one_hot)


def file_name_has_valid_coco_id(file_name, coco_ids):
    id_from_file = file_name.split("/")[-1].split(".")[0].split("_")[-1].split("-")[-1]
    return int(id_from_file) in coco_ids


def make_meta_data():  # if you want to make the meta data csv from scratch
    coco_df = pd.read_csv(
        os.path.join(DATA_DIR, "coco_meta_data.csv"), index_col="cocoId"
    )
    coco_df["categories"] = coco_df["categories"].apply(lambda x: ast.literal_eval(x))
    coco_df["supercategory"] = coco_df["supercategory"].apply(
        lambda x: ast.literal_eval(x)
    )
    coco_df["captions"] = coco_df["captions"].apply(lambda x: ast.literal_eval(x))
    nsd_df = pd.read_csv(os.path.join(DATA_DIR, "nsd_stim_info_merged.csv"))
    meta_data = coco_df.loc[nsd_df["cocoId"]]
    meta_data["nsdId"] = nsd_df.index
    meta_data["split"] = nsd_df["cocoSplit"]
    meta_data.to_csv(os.path.join(DATA_DIR, "meta_data.csv"))


def get_meta_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "meta_data.csv"), index_col="nsdId")
    df["categories"] = df["categories"].apply(lambda x: ast.literal_eval(x))
    df["supercategory"] = df["supercategory"].apply(lambda x: ast.literal_eval(x))
    df["captions"] = df["captions"].apply(lambda x: ast.literal_eval(x))
    return df
