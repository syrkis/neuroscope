# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_args, get_config, get_loaders, init_params
from src.train import train
import jax
import hydra


# main()
@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    exit()
    args, config = get_args(), get_config()
    train_loader, val_loader, _ = get_loaders(config, args)  # TODO: switch to kfold
    cnn_layer_sizes = [(3, 4, 4), (4, 8, 4)]
    mlp_layer_sizes = [8192, 256, 256, 80]
    config['layer_sizes'] = {'cnn': cnn_layer_sizes, 'mlp': mlp_layer_sizes}
    rng = jax.random.PRNGKey(0)
    params = init_params(config['layer_sizes'], rng)
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    # params, metrics = train(params, metrics, config, args, train_loader, val_loader)


# run main()
if __name__ == '__main__':
    main()