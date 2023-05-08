# main.py
#   algonauts project
# by: Noah Syrkis

# imports
from src import get_files, get_batches


# main()
def main():
    lh_file, rh_file, image_files = get_files('subj05')
    batch_loader = get_batches(lh_file, rh_file, image_files, batch_size=10)
    print(next(batch_loader))


# run main()
if __name__ == '__main__':
    main()