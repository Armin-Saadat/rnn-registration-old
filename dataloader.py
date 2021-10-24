import os
import pickle

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from path_definition import DATA_DIR


class Dataset_(Dataset):
    def __init__(self):

        # read from file
        with open(os.path.join(DATA_DIR, 'images'), 'rb') as f:
            images = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'labels'), 'rb') as f:
            labels = pickle.load(f)

        # normalize
        for i, img in enumerate(images):
            images[i] = (img / img.max()).float()
        for i, lb in enumerate(labels):
            labels[i] = (lb / lb.max()).float()

        self.images = images
        self.labels = labels
        self.image_size = images[0].shape[1:]
        self.ndims = len(self.image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        outputs = [self.images[index], self.labels[index]]
        return tuple(outputs)


def get_dataloader(batch_size, shuffle=False, pin_memory=False, workers=0):
    dataset = Dataset_()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=workers)

    return dataloader
