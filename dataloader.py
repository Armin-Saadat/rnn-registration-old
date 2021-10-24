from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Dataset_(Dataset):
    def __init__(self):

        with open('./data/images', 'rb') as f:
            images = pickle.load(f)

        with open('./data/labels', 'rb') as f:
            labels = pickle.load(f)

        self.images = images
        self.labels = labels
        self.image_size = images[0].shape[1:]
        self.ndims = len(self.image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        outputs = [self.images[index], self.labels[index]]
        return tuple(outputs)


def get_dataloader(images, labels, batch_size, shuffle=False, pin_memory=False, workers=0):
    dataset = Dataset_(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=workers)

    return dataloader
