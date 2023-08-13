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
    tmps = []
    for train_loader, _ in kfolds:
        for i in tqdm(range(1000)):
            a, b, c = next(train_loader)
            tmps.append(a)


# run main()
if __name__ == "__main__":
    main()

