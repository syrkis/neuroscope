# data.py
#     algonauts project
# by: Noah Syrkis

# imports
import os
import numpy as np


# batch_loader
def get_batches(lh_file, rh_file, image_files, batch_size):  # TODO: add option to have entire data in memory
    lh, rh = np.load(lh_file), np.load(rh_file)
    while True:
        idxs = np.random.permutation(len(image_files))
        for idx in idxs:
            image = np.load(image_files[idx])
            yield [lh[idx:idx + batch_size], rh[idx:idx + batch_size]], image
            
    