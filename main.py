# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_args, get_config, get_loaders
from src.train import train


# main()
def main():
    args, config = get_args(), get_config()
    train_loader, val_loader, _ = get_loaders(config, args)  # TODO: switch to kfold
    params, metrics = train(config, args, train_loader, val_loader)


# run main()
if __name__ == '__main__':
    main()