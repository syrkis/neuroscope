# fmri.py
#      neuroscope project
# by: Noah Syrkis

# imports
import os
import numpy as np
from nilearn import datasets
from nilearn.surface import load_surf_mesh
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

import numpy as np

def get_bold_with_coords_and_faces(challenge_vec, subject, hem, roi=None):
    """
    Return the BOLD signal data along with corresponding coordinates and faces on the brain surface.
    """
    # Get the fsaverage vector corresponding to the challenge_vec
    fsaverage_response = fsaverage_vec(challenge_vec, subject, roi, hem)

    # Determine the hemisphere to select the correct mesh
    side = "infl_left" if hem == "lh" else "infl_right"

    # Load the coordinates and faces for the selected hemisphere
    coords, faces = load_surf_mesh(ATLAS[side])

    # Create a mask for non-zero fsaverage_response
    response_mask = np.where(fsaverage_response)[0]

    # Filter the coordinates
    filtered_coords = coords[response_mask]

    # Create a mapping from old vertex indices to new ones
    index_mapping = np.full(np.max(faces) + 1, -1)  # Initialize with -1
    index_mapping[response_mask] = np.arange(response_mask.size)

    # Adjust faces to new indexing and filter out invalid faces
    filtered_faces = index_mapping[faces]
    valid_faces_mask = np.all(filtered_faces != -1, axis=1)
    filtered_faces = filtered_faces[valid_faces_mask]

    # Return the filtered coordinates, corresponding BOLD signal values, and adjusted faces
    return filtered_coords, fsaverage_response[response_mask], filtered_faces
