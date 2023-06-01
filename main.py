"""main file for neuroscope project"""""
# main.py
#   neuroscope project
# by: Noah Syrkis

# imports
from src.utils import get_args_and_config
from src.data import get_data
from src.train import train
from src.alex import run_subj
from multiprocessing import Pool


def main():
    """main function"""
    args, config = get_args_and_config()


# run main()
if __name__ == "__main__":
    main()
