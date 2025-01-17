# fmri.py
#     fmri related functions (like plotting and geometric transforms, etc.)
# by Noah  Syrkis

# Imports
import jax.numpy as jnp
import numpy as np
from nilearn import datasets, plotting
from nilearn.surface import load_surf_mesh

from neuroscope import utils

# %% Constants
ATLAS = datasets.fetch_surf_fsaverage("fsaverage")


# %% Functions
def fsa_fn(data: utils.Subject, cfg, idx):
    class_name, roi_id = roi_fn(data, cfg.roi)
    fsa = data.__getattribute__(cfg.hem).mask.fsa[class_name] == roi_id
    cha = data.__getattribute__(cfg.hem).bold[
        :, data.__getattribute__(cfg.hem).mask.cha[class_name] == roi_id
    ]
    return jnp.zeros(fsa.size).at[fsa].set(cha[idx])


# @cachier()
def mesh_fn(data, cfg, idx):
    side = "left" if cfg.hem == "lh" else "right"
    coords, faces = load_surf_mesh(ATLAS["flat_" + side])
    fsa = fsa_fn(data, cfg, idx)  # bold in fsa
    index = (-jnp.ones_like(fsa)).at[fsa != 0].set(jnp.arange(len(coords[fsa != 0])))
    faces = index[faces][np.all(index[faces] != -1, axis=1)]
    return coords[fsa != 0][:, :2], faces.astype(jnp.int32), fsa[fsa != 0]


def roi_fn(data: utils.Subject, roi: str):
    class_name = [k for k, v in data.maps.items() if roi in v.values()][0]
    roi_id = [k for k, v in data.maps[class_name].items() if v == roi][0]
    return class_name, roi_id


def plot_fn(data, cfg, idx):
    fsa = np.array(fsa_fn(data, cfg, idx))
    side = "left" if cfg.hem == "lh" else "right"
    view = plotting.view_surf(
        surf_mesh=ATLAS["pial_" + side],
        surf_map=fsa,
        bg_map=ATLAS["sulc_" + side],
        cmap="twilight_shifted",
        black_bg=True,
    )
    return view.open_in_browser()
