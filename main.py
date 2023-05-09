# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_args, get_config, get_loaders, init_params
from src.train import train
import jax


# main()
def main():
    args, config = get_args(), get_config()
    train_loader, val_loader, _ = get_loaders(config, args)  # TODO: switch to kfold
    cnn_layer_sizes = [(3, 16, 4), (16, 32, 4)]
    mlp_layer_sizes = [256, 256, 256, 80]
    config['layer_sizes'] = {'cnn': cnn_layer_sizes, 'mlp': mlp_layer_sizes}
    rng = jax.random.PRNGKey(0)
    params = init_params(config['layer_sizes'], rng)
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    params, metrics = train(params, metrics, config, args, train_loader, val_loader)


# run main()
if __name__ == '__main__':
    main()