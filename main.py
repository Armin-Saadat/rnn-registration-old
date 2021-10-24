from utils.dataloader import get_dataloader
from utils.visualizer import visualize

if __name__ == '__main__':
    batch_size = 1
    shuffle = False
    pin_memory = False
    num_workers = 0

    dataloader = get_dataloader(batch_size, shuffle, pin_memory, num_workers)
    for images, labels in dataloader:
        visualize(images, labels)


