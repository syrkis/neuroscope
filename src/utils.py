"""utility functions for neuroscope project"""""
# utils.py
#   neuroscope project
# by: Noah Syrkis

# imports
import os
import sys
import argparse
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import yaml


# PATHS
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
NSD_PATH = os.path.join(DATA_DIR, "nsd_stim_info_merged.csv")
COCO_DIR = os.path.join(DATA_DIR, "annotations")
TRAIN_CAT_FILE = os.path.join(COCO_DIR, "instances_train2017.json")
VAL_CAT_FILE = os.path.join(COCO_DIR, "instances_val2017.json")
TRAIN_CAP_FILE = os.path.join(COCO_DIR, "captions_train2017.json")
VAL_CAP_FILE = os.path.join(COCO_DIR, "captions_val2017.json")

MAKE_CATS = False
MAKE_COCO_METAS = False

ROIS = [
    "V1v",
    "V1d",
    "V2v",
    "V2d",
    "V3v",
    "V3d",
    "hV4",
    "EBA",
    "FBA-1",
    "FBA-2",
    "mTL-bodies",
    "OFA",
    "FFA-1",
    "FFA-2",
    "mTL-faces",
    "aTL-faces",
    "OPA",
    "PPA",
    "RSC",
    "OWFA",
    "VWFA-1",
    "VWFA-2",
    "mfs-words",
    "mTL-words",
    "early",
    "midventral",
    "midlateral",
    "midparietal",
    "ventral",
    "lateral",
    "parietal",
]

if MAKE_CATS:
    coco = COCO(VAL_CAT_FILE)
    cats = coco.loadCats(coco.getCatIds())
    with open(os.path.join(DATA_DIR, "coco_cats.json"), "w") as f:
        json.dump(cats, f)


with open(os.path.join(DATA_DIR, "coco_cats.json"), "r") as f:
    cat_data_for_dict = json.load(f)

# dictionaries for coco stuff and nsd stuff
cat_id_to_name = {cat["id"]: cat["name"] for cat in cat_data_for_dict}
cat_name_to_id = {cat["name"]: cat["id"] for cat in cat_data_for_dict}
coco_cat_id_to_vec_index = {cat_id: i for i, cat_id in enumerate(cat_id_to_name.keys())}
vec_index_to_coco_cat_id = {i: cat_id for i, cat_id in enumerate(cat_id_to_name.keys())}


#############
# functions #
#############


# get_nsd_files
def get_files(subject, split="training"):
    if split == "training":
        image_dir = os.path.join(DATA_DIR, subject, split + "_split", split + "_images")
        image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
        return image_files


def get_args_and_config(args_lst=None):
    subjs = 'subj01'  # ,subj02,subj03,subj04,subj05,subj07'  # skip 6 and 8 because fmri dim differs
    _rois = ",".join(ROIS)
    _roius = "EBA"
    # Load the YAML configuration file
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('config/rois.yaml', 'r') as f:
        rois = yaml.safe_load(f)

    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(f'--rois', type=str, default=_rois)
    parser.add_argument(f'--subjects', type=str, default=subjs)
    parser.add_argument(f'--n_samples', type=int, default=None)
    parser.add_argument(f'--alex', type=bool, default=False)

    # Parse the arguments and return them as a dictionary
    if 'ipykernel' in sys.modules:
        args_dict = {'rois': _rois, 'subjects': subjs, 'n_samples': 0, 'alex': True}
        args_lst = [f'--{k}={v}' for k, v in args_dict.items()]
        args = parser.parse_args(args=args_lst)
    else:
        args = parser.parse_args()

    config['n_samples'] = args.n_samples
    config['rois'] = args.rois.replace(',', ', ')
    config['subjects'] = args.subjects
    return args, config

args, config = get_args_and_config()



def extract_meta(captions_coco, instances_coco, merged_anns, nds_coco_img_ids):
    valid_ids = set(instances_coco.getImgIds())
    for img_id in tqdm(valid_ids):
        anns = captions_coco.loadAnns(captions_coco.getAnnIds(imgIds=img_id))
        captions = [ann["caption"] for ann in anns]
        innstances = instances_coco.loadAnns(instances_coco.getAnnIds(imgIds=img_id))
        categories = list(set([instance["category_id"] for instance in innstances]))
        supercategory = [
            entry["supercategory"] for entry in instances_coco.loadCats(categories)
        ]
        merged_anns.append(
            {
                "cocoId": img_id,
                "captions": captions,
                "categories": categories,
                "supercategory": supercategory,
            }
        )
    return merged_anns


if MAKE_COCO_METAS:
    train_instances_coco = COCO(TRAIN_CAT_FILE)
    val_instances_coco = COCO(VAL_CAT_FILE)
    train_captions_coco = COCO(TRAIN_CAP_FILE)
    val_captions_coco = COCO(VAL_CAP_FILE)

    merged_anns = []
    nsd_coco = pd.read_csv(NSD_PATH)
    nsd_coco_img_ids = set(nsd_coco["cocoId"])

    valid_ids = set(val_instances_coco.getImgIds()).intersection(nsd_coco_img_ids)
    merged_anns = extract_meta(
        val_captions_coco, val_instances_coco, merged_anns, nsd_coco_img_ids
    )
    merged_anns = extract_meta(
        train_captions_coco, train_instances_coco, merged_anns, nsd_coco_img_ids
    )
    df = pd.DataFrame(merged_anns)
    df.to_csv(os.path.join(DATA_DIR, "coco_meta_data.csv"), index=False)

    del (
        train_instances_coco,
        val_instances_coco,
        train_captions_coco,
        val_captions_coco,
        merged_anns,
        nsd_coco,
        df,
    )


def plot_metrics(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    axes[0].plot(metrics["train_loss"], label="train")
    axes[0].plot(metrics["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[1].plot(metrics["train_acc"], label="train")
    axes[1].plot(metrics["val_acc"], label="val")
    axes[1].set_title("accuracy")
    axes[1].legend()
    plt.show()
