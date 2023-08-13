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
from nilearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import yaml


# PATHS
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'data')
INSTANCES_DIR = os.path.join(DATA_DIR, 'coco', 'annotations', 'instances_train2017.json')
CAPTIONS_DIR = os.path.join(DATA_DIR, 'coco', 'annotations', 'captions_train2017.json')


# CONSTANTS
SUBJECTS = ['subj05', 'subj06', 'subj07', 'subj08']

CLASS_TO_ROI = {"prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
                    "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
                    "floc-faces": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
                    "floc-places": ["OPA", "PPA", "RSC"],
                    "floc-words": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
                    "streams": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]}

ROI_TO_CLASS = {roi: roi_class for roi_class, rois in CLASS_TO_ROI.items() for roi in rois}

ROIS = [roi for roi_class in CLASS_TO_ROI.values() for roi in roi_class]


# FUNCTIONS
def get_args():
    parser = argparse.ArgumentParser(description='Neuroscope Project')
    parser.add_argument('--subject', type=str, default='subj05', help='subject to use')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    return parser.parse_args()


def get_metadata_sources():
    coco_instances = COCO(INSTANCES_DIR) if 'COCO_INSTANCES' not in globals() else eval('COCO_INSTANCES')
    coco_captions = COCO(CAPTIONS_DIR) if 'COCO_CAPTIONS' not in globals() else eval('COCO_CAPTIONS')
    nsd_stim_info = pd.read_csv(os.path.join(DATA_DIR, 'nsd', 'nsd_stim_info_merged.csv'))
    return coco_instances, coco_captions, nsd_stim_info


def load_roi_data(subject):
    """load the files for the given subject"""
    roi_dir = os.path.join(DATA_DIR, 'algonauts', subject, "roi_masks")

    data = {'mapping' : {},
            'challenge' : {'lh' : {}, 'rh' : {}},
            'fsaverage' : {'lh' : {}, 'rh' : {}}}

    for roi_class in CLASS_TO_ROI.keys():
        data['mapping'][roi_class] = {'id_to_roi' : {}, 'roi_to_id' : {}}
        data['mapping'][roi_class]['id_to_roi'] = np.load(os.path.join(roi_dir, f'mapping_{roi_class}.npy'), allow_pickle=True).item()
        data['mapping'][roi_class]['roi_to_id'] = {v: k for k, v in data['mapping'][roi_class]['id_to_roi'].items()}
    
    for hem in ['lh', 'rh']:
        data['fsaverage'][hem]['all-vertices'] = np.load(os.path.join(roi_dir, f'{hem}.all-vertices_fsaverage_space.npy'))
        for roi_class in CLASS_TO_ROI.keys():
            data['challenge'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_challenge_space.npy'))
            data['fsaverage'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_fsaverage_space.npy'))

    return data