"""main file for neuroscope project"""""
# main.py
#   neuroscope project
# by: Noah Syrkis

# imports
from src.utils import get_args_and_config
from src.data import get_loaders
from src.train import train


def main():
    """main function"""
    args, config = get_args_and_config()
    folds, _ = get_loaders(args, config)
    params_lst = train(folds, config)


# run main()
if __name__ == "__main__":
    main()
