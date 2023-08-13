# main.py
#   neuroscope project
# by: Noah Syrkis


# imports
from tqdm import tqdm
from src.data import load_subject, make_kfolds


def main():
    subject = load_subject('subj05', 128)
    config = {'batch_size': 32, 'image_size': 128, 'n_splits': 5}
    kfolds = make_kfolds(subject, config)


# run main()
if __name__ == "__main__":
    main()

