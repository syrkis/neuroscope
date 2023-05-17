# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_setup, init_params
from src.data import get_loaders
from src.train import train
import jax
from tqdm import tqdm


# main function
def main():
    args, config = get_setup()
    k_fold, _ = get_loaders(args, config)
    for train_loader, val_loader in tqdm(k_fold):
        for i in range(1000):
            img, cat, sub, cap, fmri = next(train_loader)
            print(img.shape, cat.shape, sub.shape, cap.shape, fmri.shape)
            img, cat, sub, cap, fmri = next(val_loader)
            print(img.shape, cat.shape, sub.shape, cap.shape, fmri.shape)
    # rng = jax.random.PRNGKey(0)
    # params = init_params(config, rng)
    # metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # params, metrics = train(params, metrics, config, args, k_fold)


# run main()
if __name__ == "__main__":
    main()
