# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_args, get_config, get_loaders, init_params
from src.train import train
import jax


# main function
def main():
    args, config = get_args(), get_config()
    train_loader, val_loader, _ = get_loaders(args)  # TODO: switch to kfold
    rng = jax.random.PRNGKey(0)
    params = init_params(config['cnn'], rng)
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    params, metrics = train(params, metrics, config, args, train_loader, val_loader)


# run main()
if __name__ == '__main__':
    main()