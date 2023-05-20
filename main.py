"""main file for neuroscope project"""""
# main.py
#   neuroscope project
# by: Noah Syrkis

# imports
import jax
import haiku as hk
import optax
from tqdm import tqdm
from src import get_setup
from src.data import get_loaders
from src.train import train


def main():
    """main function"""
    args, config = get_setup()
    k_fold, _ = get_loaders(args, config)
    params = train(k_fold, config)


# run main()
if __name__ == "__main__":
    main()
