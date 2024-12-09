# utils.py
#   neuroscope project
# by: Noah Syrkis


# imports
import os
from chex import dataclass
from jaxtyping import Array
from typing import Mapping


@dataclass
class Mask:
    fsa: Mapping[str, Array]
    cha: Mapping[str, Array]


@dataclass
class Hemisphere:
    mask: Mask
    bold: Array


@dataclass
class COCO:
    imgs: Array
    meta: Array


@dataclass
class Subject:
    rh: Hemisphere
    lh: Hemisphere
    maps: Mapping[str, Mapping[int, str]]
    coco: COCO
    # coco: None = None


# CONFIG
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "data")

# CLASS_TO_ROI = {
# "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
# "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
# "floc-faces": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
# "floc-places": ["OPA", "PPA", "RSC"],
# "floc-words": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
# "streams": [
# "early",
# "midventral",
# "midlateral",
# "midparietal",
# "ventral",
# "lateral",
# "parietal",
# ],
# }

# ROI_TO_CLASS = {
# roi: roi_class for roi_class, rois in CLASS_TO_ROI.items() for roi in rois
# }


# model stuff
# def get_args():
#     parser = argparse.ArgumentParser(description="Neuroscope Project")
#     parser.add_argument("--subject", type=str, default="subj05", help="subject to use")
#     parser.add_argument("--image_size", type=int, default=256, help="image size")
#     return parser.parse_args()
