# streamlit_app.py
#     neuroscope project
# by: Noah Syrkis

# imports
import os
import numpy as np
import streamlit.components.v1 as components
from tqdm import tqdm
from src.utils import DATA_DIR, SUBJECTS



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
        black_bg=True,
    )
    view_html = view.get_iframe().to_html()
    components.html(view_html, height=800, width=1200)

# call plot_brain
plot_brain(challenge_vec, subject, hem, roi=None)