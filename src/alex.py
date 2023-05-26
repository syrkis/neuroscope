# alex.py
#     neuroscope project alexnet baseline
# by: Noah Syrkis

# imports
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from PIL import Image
from tqdm import tqdm
from src.utils import get_args_and_config, get_files
from src.data import preprocess


# main
def run_alex():
    args, config = get_args_and_config()
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
    model.eval()
    for subject in args.subjects.split(","):
        run_subject(subject, model, config)

def get_imgs(subject, config):
    """return a list of images for a subject"""
    img_files = [f for f in get_files(subject)][: config['n_samples']]
    imgs = [preprocess(Image.open(f), config['image_size']) for f in tqdm(img_files)]
    imgs = [torch.from_numpy(img) for img in imgs]  # to torch tensor
    imgs = [img.permute(2, 0, 1) for img in imgs]  # put channel first
    return imgs

def run_subject(subject, model, config):
    """run alexnet on a subject"""
    imgs = get_imgs(subject, config)
    for img in imgs:
        output = model(img)
        print(output)
        break