from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck

if __name__ == '__main__':

    device = 'cpu'
    batch_size = 1
    shuffle = False
    pin_memory = False
    num_workers = 0

    dataloader = get_dataloader(batch_size, shuffle, pin_memory, num_workers)
    model = Bottleneck(dataloader.dataset.image_size)

    for images, labels in dataloader:
        images = images.permute(1, 0, 2, 3).unsqueeze(2).to(device)
        labels = labels.permute(1, 0, 2, 3).unsqueeze(2).to(device)
        model(images)
