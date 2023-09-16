# fmri.py
#      neuroscope project
# by: Noah Syrkis

# imports
import os
import numpy as np
from nilearn import datasets
from src.utils import SUBJECTS, ROI_TO_CLASS, load_roi_data


# globals
ROI_DATA = {subject: load_roi_data(subject) for subject in SUBJECTS}
ATLAS = datasets.fetch_surf_fsaverage("fsaverage")


# functions
def get_space_mappings(subject: str, hem: str, roi: str) -> np.ndarray:
    """return the fsaverage space mapping for the given subject and hemisphere"""
    if not roi:
        fsaverage_space = ROI_DATA[subject]['fsaverage'][hem]['all-vertices']
        return fsaverage_space
    else:
        fsaverage_space = ROI_DATA[subject]['fsaverage'][hem][ROI_TO_CLASS[roi]]
        challenge_space = ROI_DATA[subject]['challenge'][hem][ROI_TO_CLASS[roi]]
        roi_id = ROI_DATA[subject]['mapping'][ROI_TO_CLASS[roi]]['roi_to_id'][roi]
        fsaverage_space = np.asarray(fsaverage_space == roi_id, dtype=int)
        challenge_space = np.asarray(challenge_space == roi_id, dtype=int)
        return fsaverage_space, challenge_space


def fsaverage_vec(challenge_vec, subject, roi, hem) -> np.ndarray:
    """convert a challenge vector to fsaverage space"""
    if roi:
        fsaverage_space, challenge_space = get_space_mappings(subject, hem, roi)
        fsaverage_response = np.zeros(len(fsaverage_space))
        fsaverage_response[np.where(fsaverage_space)[0]] = \
        challenge_vec[np.where(challenge_space)[0]]
    else:
        fsaverage_space = get_space_mappings(subject, hem, roi)
        fsaverage_response = np.zeros(len(fsaverage_space))
        fsaverage_response[np.where(fsaverage_space)[0]] = challenge_vec
    return fsaverage_response



