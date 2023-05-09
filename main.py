# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_args, get_config, get_loaders


# main()
def main():
    args = get_args()
    config = get_config()
    train_loader, val_loader, test_loader = get_loaders(config, args)
    img, cat, sup, cap = next(train_loader)
    print(img.shape, cat.shape, sup.shape, cap.shape)
    


# run main()
if __name__ == '__main__':
    main()