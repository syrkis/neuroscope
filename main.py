"""main file for neuroscope project"""""
# main.py
#   neuroscope project
# by: Noah Syrkis

# imports
import wandb
from multiprocessing import Pool
from src.utils import get_args_and_config
from src.data import get_data
from src.train import sweep, train, test
from src.eval import test


def main():
    """main function"""
    args, config, params = get_args_and_config()

    if args.sweep:
        data = get_data(args)
        sweep(data, config)

    if args.train:
        data = get_data(args)
        train(data, params)

    if args.test:
        data = get_data(args)
        test_data = {s: d for s, (_, d) in data.items()}
        test(test_data, config)


# run main()
if __name__ == "__main__":
    main()
