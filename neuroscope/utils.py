# utils.py
#   neuroscope project
# by: Noah Syrkis


# imports
from chex import dataclass
from jaxtyping import Array, PyTree
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


@dataclass
class Config:
    cnn_d = [1, 8, 16]
    mlp_d = [4096, 100]
    batch_size = 64
    image_size = 64
    stride = (2, 2)
    lr = 0.001
    subj = "subj01"
    roi = "V2d"
    hem = "lh"
    epochs = 200


@dataclass
class Module:
    mlp: PyTree[Array]
    cnn: PyTree[Array]


@dataclass
class Params:
    encode: Module
    decode: Module
