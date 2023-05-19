"""main file for neuroscope project"""""
# main.py
#   neuroscope project
# by: Noah Syrkis

# imports
import jax
from src import get_setup, init_params
from src.data import get_loaders
from src.train import train


def main():
    """main function"""
    args, config = get_setup()
    k_fold, _ = get_loaders(args, config)
    train_loader, val_loader = next(k_fold)
    # rng = jax.random.PRNGKey(0)
    # params = init_params(config, rng)
    # metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # params, metrics = train(params, metrics, config, args, k_fold)


# run main()
if __name__ == "__main__":
    main()
