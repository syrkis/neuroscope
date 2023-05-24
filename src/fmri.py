"""fmri data processing"""""
# fmri.py
#      neuroscope project
# by: Noah Syrkis

# imports
import os
import numpy as np
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
import networkx as nx


# constants
atlas = datasets.fetch_surf_fsaverage("fsaverage")
DATA_DIR = "data/"


# load fmri data
def get_fmri(subject: str) -> tuple:
    """return the fmri data for the given subject"""
    lh = np.load(fmri_data(subject, "l"))
    rh = np.load(fmri_data(subject, "r"))
    return lh, rh

# connectome construction
def get_connectome(subject: str, roi: str, hem: str) -> np.ndarray:
    """given a roi, return the connectome for that roi in the given hemisphere"""
    lh, rh = get_fmri(subject)
    roi_mask = get_roi_mask(roi, hem, atlas)
    return lh[:, roi_mask] if hem == "left" else rh[:, roi_mask]


def fmri_data(subject: str, hem: str) -> str:
    """given a hemisphere, return the fmri data directory"""
    return os.path.join(
        DATA_DIR, subject, "training_split",
        "training_fmri", hem + "h_training_fmri.npy"
                        )

def get_roi_dir(subject):
    return os.path.join(DATA_DIR, subject, "roi_masks")

def get_mapping_files(subject):
    roi_dir = get_roi_dir(subject)
    return [
        os.path.join(roi_dir, f)
        for f in sorted(os.listdir(roi_dir))
        if f.startswith("mapping_")
    ]

def get_challenge_files(subject):
    roi_dir = get_roi_dir(subject)
    return [
        os.path.join(roi_dir, f)
        for f in sorted(os.listdir(roi_dir))
        if f.endswith("challenge_space.npy")
    ]

def get_fsaverage_files(subject):
    roi_dir = get_roi_dir(subject)
    return [
        os.path.join(roi_dir, f)
        for f in sorted(os.listdir(roi_dir))
        if f.endswith("fsaverage_space.npy")
    ]

def get_mappings(subject):
    mapping_files = get_mapping_files(subject)
    return {
        f.split("/")[-1].split("_")[1].split(".")[0]: np.load(f, allow_pickle=True).item()
        for f in mapping_files
    }

def get_challenge(subject):
    challenge_files = get_challenge_files(subject)
    return {
        f.split("/")[-1].split("_")[0]: np.load(f) for f in challenge_files
    }

def get_fsaverage(subject):
    fsaverage_files = get_fsaverage_files(subject)
    return {
        f.split("/")[-1].split("_")[0]: np.load(f) for f in fsaverage_files
    }



def roi_to_roi_class(subject, roi):
    """given a roi, return the roi class it belongs to"""
    mappings = get_mappings(subject)
    for k, v in mappings.items():
        if roi in v.values():
            return k
    return None


def roi_to_map_index(subject, roi):
    """given a roi, return the index of the roi mapping within the roi class"""
    mappings = get_mappings(subject) 
    roi_class = roi_to_roi_class(subject, roi)
    roi_map = mappings[roi_class]
    return {v: k for k, v in roi_map.items()}[
        roi
    ]  # confusing that idx is used in mulitple classes


def get_roi_mask(subject, roi, hem, atlas="challenge"):
    """given a roi, return the roi mask for the given hemisphere"""
    challenge = get_challenge(subject)
    fsaverage = get_fsaverage(subject)
    roi_class = roi_to_roi_class(subject, roi)
    roi_mapping = roi_to_map_index(subject, roi)
    atlas = challenge if atlas == "challenge" else fsaverage
    return atlas[f"{hem[0]}h.{roi_class}"] == roi_mapping


def get_multi_roi_mask(subject, rois, hem, atlas="challenge"):
    """given a roi, return the roi mask for the given hemisphere"""
    challenge = get_challenge(subject)
    roi_mask = np.zeros(len(challenge[hem[0] + "h.floc-bodies"])).astype(bool)
    for roi in rois.split(","):
        roi_mask += get_roi_mask(subject, roi, hem, atlas)
    return roi_mask


def roi_response_to_image(subject, roi, idxs, hem):  # TODO: ensure correctness
    """given a roi, return the response to image for that roi in the given hemisphere"""
    lh_fmri, rh_fmri = get_fmri(subject)
    roi_mask = get_roi_mask(subject, roi, hem, atlas="challenge")
    if hem == "left":
        res = lh_fmri[:, roi_mask][idxs]
    if hem == "right":
        res = rh_fmri[:, roi_mask][idxs]
    return res


def fsaverage_roi(subject, roi, hem):  # TODO: ensure correctness
    """given a roi, return the roi mask for the given hemisphere"""
    fsaverage = get_fsaverage(subject)
    roi_class = roi_to_roi_class(subject, roi)
    fsaverage_roi_mask = fsaverage[f"{hem[0]}h.{roi_class}"] == roi_to_map_index(roi)
    return fsaverage_roi_mask.astype(int)


def fsaverage_roi_response_to_image(subject, roi, img, hem):  # TODO: ensure correctness
    """given a roi, return the response to image for that roi in the given hemisphere"""
    lh_fmri, rh_fmri = get_fmri(subject)
    fsaverage_roi_mask = get_roi_mask(subject, roi, hem, atlas="fsaverage")
    challenge_roi_mask = get_roi_mask(subject, roi, hem, atlas="challenge")
    res = np.zeros(len(fsaverage_roi_mask))
    if hem == "left":
        res[np.where(fsaverage_roi_mask)[0]] = lh_fmri[
            img, np.where(challenge_roi_mask)[0]
        ]
    if hem == "right":
        res[np.where(fsaverage_roi_mask)[0]] = rh_fmri[
            img, np.where(challenge_roi_mask)[0]
        ]
    return res


def connectome_from_roi_response(subject, roi, hem):  # this is wrong
    """given a roi, return the connectome for that roi in the given hemisphere"""
    lh_fmri, rh_fmri = get_fmri(subject)
    roi_mask = get_roi_mask(subject, roi, hem, atlas="challenge")
    fmri = lh_fmri if hem == "left" else rh_fmri
    roi_response = fmri[:, roi_mask]
    connectivity_measure = ConnectivityMeasure(kind="covariance")
    connectivity_matrix = connectivity_measure.fit_transform([roi_response])[0]
    connectome = connectivity_matrix_to_connectome(connectivity_matrix)
    return connectome


def connectivity_matrix_to_connectome(connectivity_matrix):
    """given a connectivity matrix, return a graph"""
    N = connectivity_matrix.shape[0]
    thresh = np.percentile(
        np.abs(connectivity_matrix), 100 * (N - (N / 100)) / N
    )  # consider thresholding differently as n edges increases with nodes ** 2
    connectivity_matrix[np.abs(connectivity_matrix) < thresh] = 0
    # set diagonal to 0
    np.fill_diagonal(connectivity_matrix, 0)
    graph = nx.from_numpy_array(connectivity_matrix)
    return graph, connectivity_matrix

# TODO: add a function to get the connectome for a given roi for a particular image