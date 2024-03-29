# plots.py
#     neuroscope plots
# by: Noah Syrkis

# imports
import numpy as np
from IPython.display import display, Image, clear_output
import matplotlib.pyplot as plt
import imageio
import networkx as nx
from nilearn import plotting
from tqdm import tqdm
from IPython.display import display, HTML
import time
import numpy as np
import base64
from PIL import Image as PILImage
from io import BytesIO
from jinja2 import Template, Environment, FileSystemLoader
from jax import vmap
import darkdetect
from src.fmri import ATLAS, fsaverage_vec
from src.utils import matrix_to_image, CONFIG


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
