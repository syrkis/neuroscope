# A Multi-Modal Exploration of the Algonauts Dataset

This repository explores the connection between visual stimuli and brain response.
It is based on the [Algonauts](https://algonauts.csail.mit.edu/) dataset, which contains fMRI data from 8 subjects,
each reacting to thousands of images. The Algonauts dataset, in turn is based on the the [NSD](https://naturalscenesdataset.org/) dataset,
which uses images from [COCO](https://cocodataset.org/#home). COCO image category, supercategory, and captions are used in this project,
in addition to the pure Algonauts data.

Tools of note include JAX and nilearn, in addition to the usual suspects (numpy, pandas, matplotlib, etc.).
`src` contains the main code, `notebooks` contains the main notebooks, and `data` contains the data
(which is not included in this repository). `src` is further divided into `data`, `models`, `train`, and `utils`
for data loading, model definitions, training, and utility functions, respectively.

To run the project locally, please ensure that `data` is populated with the Algonauts data, and our (currently unavailable) COCO augmentations.
Then, run `pip install -r requirements.txt` to install the necessary dependencies in a virtual python 3.11 environment.