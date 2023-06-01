# alex.py
#     neuroscope project alexnet baseline
# by: Noah Syrkis

# imports
# imports
import warnings; warnings.filterwarnings('ignore')
import os
import torch
from multiprocessing import Pool
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
from PIL import Image
from src.utils import DATA_DIR

# get data and model
BATCH_SIZE = 500
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
feature_extractor = create_feature_extractor(model, return_nodes=["features.2"])

# main
def get_img_files(subj):
    subj_img_dir = os.path.join(DATA_DIR, subj, 'training_split/training_images')
    subj_img_files = [os.path.join(subj_img_dir, f) for f in os.listdir(subj_img_dir) if f.endswith('.png')]
    return sorted(subj_img_files)

def load_img_files(subj):
    # images are pngs
    img_files = get_img_files(subj)
    imgs = []
    for f in tqdm(img_files):  # make sure not to have too many files open
        with Image.open(f) as img:
            img = img.convert('RGB')
            imgs.append(img)
    imgs = [img.resize((224, 224)) for img in imgs]
    imgs = [torch.from_numpy(np.array(img)) for img in imgs]
    imgs = torch.stack(imgs)
    imgs = imgs / 255.0
    imgs = imgs.permute(0, 3, 1, 2)
    imgs = normalize(imgs)
    return imgs

def normalize(imgs):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    imgs = imgs.float()
    for i in range(3):
        imgs[:, i, :, :] = (imgs[:, i, :, :] - means[i]) / stds[i]
    return imgs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

def fit_pca(feature_extractor, loader):
    pca = IncrementalPCA(n_components=100, batch_size=BATCH_SIZE)
    for batch in tqdm(loader):
        with torch.no_grad():
            features = feature_extractor(batch)
            features = torch.hstack([torch.flatten(l, start_dim=1) for l in features.values()])
            pca.partial_fit(features)
    return pca

def extract_features(feature_extractor, loader, pca):
    features = []
    for batch in loader:
        with torch.no_grad():
            batch_features = feature_extractor(batch)
            batch_features = torch.hstack([torch.flatten(l, start_dim=1) for l in batch_features.values()])
            batch_features = pca.transform(batch_features)
            features.append(batch_features)
    features = np.vstack(features)
    return features

def run_subj(subj):
    ds = Dataset(load_img_files(subj))
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    pca = fit_pca(feature_extractor, dl)
    features = extract_features(feature_extractor, dl, pca)
    np.save(os.path.join(DATA_DIR, subj, 'training_split', 'alexnet_features.npy'), features)