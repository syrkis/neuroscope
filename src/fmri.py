# fmri.py
#      neuroscope project
# by: Noah Syrkis

# imports
import jax
from jax import numpy as jnp
from jax import random, grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import os
from nilearn import datasets, plotting


# connectome construction
def get_connectome(roi, hem, atlas="challenge"):
    # given a roi, return the connectome for that roi in the given hemisphere
    roi_mask = get_roi_mask(roi, hem, atlas)
    if hem == "left":
        res = lh_fmri[:, roi_mask]
    if hem == "right":
        res = rh_fmri[:, roi_mask]
    return res


# constants
subject = "subj05"
data_dir = "data/"
fmri_data = lambda h: os.path.join(
    data_dir, subject, "training_split", "training_fmri", h + "h_training_fmri.npy"
)
roi_dir = os.path.join(data_dir, subject, "roi_masks")
mapping_files = [
    os.path.join(roi_dir, f)
    for f in sorted(os.listdir(roi_dir))
    if f.startswith("mapping_")
]
challenge_files = [
    os.path.join(roi_dir, f)
    for f in sorted(os.listdir(roi_dir))
    if f.endswith("challenge_space.npy")
]
fsaverage_files = [
    os.path.join(roi_dir, f)
    for f in sorted(os.listdir(roi_dir))
    if f.endswith("fsaverage_space.npy")
]

lh_fmri, rh_fmri = np.load(fmri_data("l")), np.load(fmri_data("r"))
mappings = {
    f.split("/")[-1].split("_")[1].split(".")[0]: np.load(f, allow_pickle=True).item()
    for f in mapping_files
}
challenge = {f.split("/")[-1].split("_")[0]: np.load(f) for f in challenge_files}
fsaverage = {f.split("/")[-1].split("_")[0]: np.load(f) for f in fsaverage_files}
atlas = datasets.fetch_surf_fsaverage("fsaverage")


# functions  # TODO: make conectome creator function (current setup representes res as flattened array)
def roi_to_roi_class(roi):
    # given a roi return the roi class it belongs to
    for k, v in mappings.items():
        if roi in v.values():
            return k
    return None


def roi_to_map_index(roi):
    # given a roi, return the index of the roi mapping within the roi class
    roi_class = roi_to_roi_class(roi)
    roi_map = mappings[roi_class]
    return {v: k for k, v in roi_map.items()}[
        roi
    ]  # confusing that idx is used in mulitple classes


def roi_to_roi_class(roi):
    # given a roi return the roi class it belongs to
    for k, v in mappings.items():
        if roi in v.values():
            return k
    return None


def get_roi_mask(roi, hem, atlas="challenge"):
    roi_class = roi_to_roi_class(roi)
    roi_mapping = roi_to_map_index(roi)
    if atlas == "challenge":
        roi_mask = challenge[f"{hem[0]}h.{roi_class}"] == roi_mapping
    if atlas == "fsaverage":
        roi_mask = fsaverage[f"{hem[0]}h.{roi_class}"] == roi_mapping
    return roi_mask


def get_multi_roi_mask(rois, hem, atlas="challenge"):
    roi_mask = np.zeros(len(challenge[hem[0] + "h.floc-bodies"])).astype(bool)
    for roi in rois.split(","):
        roi_mask += get_roi_mask(roi, hem, atlas)
    return roi_mask


def roi_response_to_image(roi, idxs, hem):  # TODO: ensure correctness
    # given a roi, return the response to image for that roi in the given hemisphere
    roi_mask = get_roi_mask(roi, hem, atlas="challenge")
    if hem == "left":
        res = lh_fmri[:, roi_mask][idxs]
    if hem == "right":
        res = rh_fmri[:, roi_mask][idxs]
    return res


def fsaverage_roi(roi, hem):
    # given a roi, return the roi mask for the fsaverage brain
    roi_class = roi_to_roi_class(roi)
    fsaverage_roi_mask = fsaverage[f"{hem[0]}h.{roi_class}"] == roi_to_map_index(roi)
    return fsaverage_roi_mask.astype(int)


def fsaverage_roi_response_to_image(roi, img, hem):  # TODO: ensure correctness
    # given a roi, return the response to image for that roi in the given hemisphere
    fsaverage_roi_mask = get_roi_mask(roi, hem, atlas="fsaverage")
    challenge_roi_mask = get_roi_mask(roi, hem, atlas="challenge")
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


def plot_region(roi, hem, img=None):
    # if variable called view exists, close it
    if img is None:
        surface = fsaverage_roi(roi, hem)
    else:
        surface = fsaverage_roi_response_to_image(roi, img, hem)
    view = plotting.view_surf(
        surf_mesh=atlas["pial_" + hem],
        surf_map=surface,
        bg_map=atlas["sulc_" + hem],
        threshold=1e-14,
        cmap="twilight_r",
        colorbar=True,
        title=roi + ", " + hem + " hemisphere",
    )
    view.open_in_browser()
