# utils.py
#   algonauts project
# by: Noah Syrkis

# imports
import os
from argparse import ArgumentParser
import json
from pycocotools.coco import COCO


# PATHS
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
NSD_PATH = os.path.join(DATA_DIR, 'nsd_stim_info_merged.csv')
COCO_DIR = os.path.join(DATA_DIR, 'annotations')
TRAIN_CAT_FILE = os.path.join(COCO_DIR, 'instances_train2017.json')
VAL_CAT_FILE = os.path.join(COCO_DIR, 'instances_val2017.json')
TRAIN_CAP_FILE = os.path.join(COCO_DIR, 'captions_train2017.json')
VAL_CAP_FILE = os.path.join(COCO_DIR, 'captions_val2017.json')

make_cats = False

if make_cats:
    coco = COCO(VAL_CAT_FILE)
    cats = coco.loadCats(coco.getCatIds())
    with open(os.path.join(DATA_DIR, 'coco_cats.json'), 'w') as f:
        json.dump(cats, f)


with open(os.path.join(DATA_DIR, 'coco_cats.json'), 'r') as f:
    cat_data_for_dict = json.load(f)

# dictionaries for coco stuff and nsd stuff
cat_id_to_name = {cat['id']: cat['name'] for cat in cat_data_for_dict}
cat_name_to_id = {cat['name']: cat['id'] for cat in cat_data_for_dict}
coco_cat_id_to_vec_index = {cat_id: i for i, cat_id in enumerate(cat_id_to_name.keys())}
vec_index_to_coco_cat_id = {i: cat_id for i, cat_id in enumerate(cat_id_to_name.keys())}


#############
# functions #
############# 

# get_nsd_files
def get_files(subject, split='training'):
    if split == 'training':
        lh_fmri_file = os.path.join(DATA_DIR, subject, 'training_split/training_fmri/lh_training_fmri.npy')
        rh_fmri_file = os.path.join(DATA_DIR, subject, 'training_split/training_fmri/rh_training_fmri.npy')
    image_dir = os.path.join(DATA_DIR, subject, split + '_split', split + '_images')
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    return lh_fmri_file, rh_fmri_file, image_files


# get command line arguments
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--subject', type=str, default='subj05')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--n', type=int, default=100)
    return parser.parse_args()

# get config file
def get_config():
    with open(os.path.join(ROOT_DIR, 'config.json')) as f:
        config = json.load(f)
        config['layer_sizes'] = [config['image_size'] ** 2] + config['hidden_layer_sizes'] + [len(cat_id_to_name)]
    return config