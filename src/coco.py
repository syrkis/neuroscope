"""functions for getting coco meta data for the algonauts subset"""
# coco.py
#   coco data set stuff
# by: Noah Syrkis

# imports
import os
import ast
from jax import numpy as jnp
import numpy as np
import pandas as pd
from src.utils import DATA_DIR, cat_id_to_name, coco_cat_id_to_vec_index


# functions for coco data sets, including loading and preprocessing, making meta data files, etc.
def preprocess(image: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Image preprocessing function."""
    image = image.resize((image_size, image_size))
    image = np.array(image)
    image = image / 255.0
    image = image.astype(jnp.float32)
    return image


def c_to_one_hot(categories: list) -> jnp.ndarray:
    """Converts a list of categories to a one hot vector."""
    # there are 80 categories, but the ids range from 1-90 (skipping 10)
    one_hot = np.zeros(len(cat_id_to_name))
    for cat in categories:
        one_hot[coco_cat_id_to_vec_index[cat]] = 1
    return jnp.array(one_hot)


def file_name_has_valid_coco_id(file_name: str, coco_ids: set) -> bool:
    """test to see if a file name has a valid coco id"""
    id_from_file = file_name.split("/")[-1].split(".")[0].split("_")[-1].split("-")[-1]
    return int(id_from_file) in coco_ids


def make_meta_data():  # if you want to make the meta data csv from scratch
    """function makes and saves coco meta data dataframe for the algonatuts subset"""
    coco_df = pd.read_csv(
        os.path.join(DATA_DIR, "coco_meta_data.csv"), index_col="cocoId"
    )
    coco_df["categories"] = coco_df["categories"].apply(ast.literal_eval)
    coco_df["supercategory"] = coco_df["supercategory"].apply(ast.literal_eval)
    coco_df["captions"] = coco_df["captions"].apply(ast.literal_eval)
    nsd_df = pd.read_csv(os.path.join(DATA_DIR, "nsd_stim_info_merged.csv"))
    meta_data = coco_df.loc[nsd_df["cocoId"]]
    meta_data["nsdId"] = nsd_df.index
    meta_data["split"] = nsd_df["cocoSplit"]
    meta_data.to_csv(os.path.join(DATA_DIR, "meta_data.csv"))


def get_meta_data() -> pd.DataFrame:
    """get the created meta data dataframe"""
    df = pd.read_csv(os.path.join(DATA_DIR, "meta_data.csv"), index_col="nsdId")
    df["categories"] = df["categories"].apply(ast.literal_eval)
    df["supercategory"] = df["supercategory"].apply(ast.literal_eval)
    df["captions"] = df["captions"].apply(ast.literal_eval)
    return df
