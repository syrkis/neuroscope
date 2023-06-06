"""fmri data processing"""""
# fmri.py
#      neuroscope project
# by: Noah Syrkis

# imports
import os
import numpy as np
from nilearn import datasets
from nilearn import plotting
from tqdm import tqdm
from src.utils import DATA_DIR, SUBJECTS

# globals
atlas = datasets.fetch_surf_fsaverage("fsaverage")

roi_class_to_roi = {"prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
                    "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
                    "floc-faces": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
                    "floc-places": ["OPA", "PPA", "RSC"],
                    "floc-words": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
                    "streams": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]}

roi_to_roi_class = {roi: roi_class for roi_class, rois in roi_class_to_roi.items() for roi in rois}


#  load roi data
def load_roi_data(subject):
    """load the files for the given subject"""
    roi_dir = os.path.join(DATA_DIR, subject, "roi_masks")

    data = {'mapping' : {},
            'challenge' : {'lh' : {}, 'rh' : {}},
            'fsaverage' : {'lh' : {}, 'rh' : {}}}

    for roi_class in roi_class_to_roi.keys():
        data['mapping'][roi_class] = {'id_to_roi' : {}, 'roi_to_id' : {}}
        data['mapping'][roi_class]['id_to_roi'] = np.load(os.path.join(roi_dir, f'mapping_{roi_class}.npy'), allow_pickle=True).item()
        data['mapping'][roi_class]['roi_to_id'] = {v: k for k, v in data['mapping'][roi_class]['id_to_roi'].items()}
    
    for hem in ['lh', 'rh']:
        data['fsaverage'][hem]['all-vertices'] = np.load(os.path.join(roi_dir, f'{hem}.all-vertices_fsaverage_space.npy'))
        for roi_class in roi_class_to_roi.keys():
            data['challenge'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_challenge_space.npy'))
            data['fsaverage'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_fsaverage_space.npy'))

    return data

def load_fmri(subject):
    """load the fmri data for the given subject"""
    lh_fmri_file = os.path.join(DATA_DIR, subject, "training_split", "training_fmri", "lh_training_fmri.npy")
    rh_fmri_file = os.path.join(DATA_DIR, subject, "training_split", "training_fmri", "rh_training_fmri.npy")
    lh = np.load(lh_fmri_file)
    rh = np.load(rh_fmri_file)
    data = {'lh' : lh, 'rh' : rh}
    return data


roi_data = {subject : load_roi_data(subject) for subject in tqdm(SUBJECTS)}
fmri_data = {subject : load_fmri(subject) for subject in tqdm(SUBJECTS)}


# vector toi roi (image or correlation)
def spaces(subject: str, hem: str, roi: str) -> np.ndarray:
    """return the fsaverage space mapping for the given subject and hemisphere"""
    if not roi:
        fsaverage_space = roi_data[subject]['fsaverage'][hem]['all-vertices']
        return fsaverage_space
    else:
        fsaverage_space = roi_data[subject]['fsaverage'][hem][roi_to_roi_class[roi]]
        challenge_space = roi_data[subject]['challenge'][hem][roi_to_roi_class[roi]]
        roi_id = roi_data[subject]['mapping'][roi_to_roi_class[roi]]['roi_to_id'][roi]
        fsaverage_space = np.asarray(fsaverage_space == roi_id, dtype=int)
        challenge_space = np.asarray(challenge_space == roi_id, dtype=int)
        return fsaverage_space, challenge_space


def fsaverage_vec(challenge_vec, subject, roi):
    """convert a challenge vector to fsaverage space"""
    hem = "lh" if challenge_vec.shape[0] == 19004 else "rh"  # r might have wrong dimensions
    if roi:
        fsaverage_space, challenge_space = spaces(subject, hem, roi)
        fsaverage_response = np.zeros(len(fsaverage_space))
        fsaverage_response[np.where(fsaverage_space)[0]] = \
        challenge_vec[np.where(challenge_space)[0]]
    else:
        fsaverage_space = spaces(subject, hem, roi)
        fsaverage_response = np.zeros(len(fsaverage_space))
        fsaverage_response[np.where(fsaverage_space)[0]] = challenge_vec
    return fsaverage_response


def plot_brain(challenge_vec, subject, hem, roi=None):
    """plot a vector on the brain"""
    surface = fsaverage_vec(challenge_vec, subject, roi)
    direction = "left" if hem == "lh" else "right"
    title = direction + " hemisphere, " + "subject " + subject[-1]
    title = title + ", " + roi if roi else title
    view = plotting.view_surf(
        surf_mesh=atlas["pial_" + direction],
        surf_map=surface,
        bg_map=atlas["sulc_" + direction],
        threshold=1e-14,
        cmap="twilight_shifted",
        colorbar=True,
        title=title,
    )
    return view.resize(height=400, width=1200)