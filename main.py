"""main file for neuroscope project"""""
# main.py
#   neuroscope project
# by: Noah Syrkis

# imports
from src.utils import get_args_and_config
from src.data import get_data
from src.train import train
from multiprocessing import Pool


def main():
    """main function"""
    args, config = get_args_and_config()
    data = get_data(args, config)


# run main()
if __name__ == "__main__":
    main()
