# plots.py
#     neuroscope plots
# by: Noah Syrkis

# imports
import numpy as np
import matplotlib.pyplot as plt
import imageio
import networkx as nx
from nilearn import plotting
from tqdm import tqdm
from src.fmri import ATLAS, fsaverage_vec


# globals
plt.style.use("dark_background")


# functions
def plot_brain(challenge_vec, subject, hem, roi=None):
    """plot a vector on the brain"""
    surface = fsaverage_vec(challenge_vec, subject, roi, hem)
    side = "left" if hem == "lh" else "right"
    title = side + " hemisphere, " + "subject " + subject[-1]
    title = title + ", " + roi if roi else title
    view = plotting.view_surf(
        surf_mesh=ATLAS["pial_" + side],
        surf_map=surface,
        bg_map=ATLAS["sulc_" + side],
        threshold=1e-14,
        cmap="twilight_shifted",
        colorbar=True,
        title=title,
        black_bg=True,
    )
    return view.resize(height=900, width=1200)

# plot decodings
def plot_decoding(decodings, n=4):
    """small multiple gifs of decodings at differnt stages of training"""
    decodings = decodings[:n * n]
    fig, axs = plt.subplots(n, n, figsize=(n * 2, n * 2))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(decodings[i])
        ax.axis("off")
    plt.tight_layout()
    plt.close()
    return fig


def plot_decoding_progress():
    pass