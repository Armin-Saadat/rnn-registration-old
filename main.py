import os
import pickle

from path_definition import DATA_DIR
from dataloader import get_dataloader

if __name__ == '__main__':
    batch_size = 1
    shuffle = False
    pin_memory = False
    num_workers = 0

    dataloader = get_dataloader(batch_size, shuffle, pin_memory, num_workers)
    for images, labels in dataloader:
        print(images.shape)
        with open(os.path.join(DATA_DIR, 'images_check'), 'wb') as f:
            pickle.dump(images, f)

        print(labels.shape)
        with open(os.path.join(DATA_DIR, 'labels_check'), 'wb') as f:
            pickle.dump(labels, f)

        exit()
