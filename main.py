"""main file for neuroscope project"""""
# main.py
#   neuroscope project
# by: Noah Syrkis

# imports
from src.utils import get_args_and_config
from src.data import get_data
from src.train import train
from src.alex import run_alex


def main():
    """main function"""
    run_alex()
    # args, config = get_args_and_config()
    # data = get_data(args, config)
    # params_lst = train(folds, config)


# run main()
if __name__ == "__main__":
    main()
