# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_setup, init_params
from src.data import get_loaders
from src.train import train
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt


# main function
def main():
    args, config = get_setup()
    k_fold, _ = get_loaders(args, config)
    # rng = jax.random.PRNGKey(0)
    # params = init_params(config, rng)
    # metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # params, metrics = train(params, metrics, config, args, k_fold)


# run main()
if __name__ == "__main__":
    main()
