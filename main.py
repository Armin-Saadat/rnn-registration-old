import argparse

from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck


def run(args):
    dataloader = get_dataloader(args.batch_size, args.shuffle, args.pin_memory, args.num_workers)
    model = Bottleneck(dataloader.dataset.image_size).to(args.device)
    model.train()

    for images, labels in dataloader:
        images = images.permute(1, 0, 2, 3).unsqueeze(2).to(args.device)
        labels = labels.permute(1, 0, 2, 3).unsqueeze(2).to(args.device)
        model(images, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-lr', type=float, default=0.001, help='learning-rate')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-smooth_weight', type=float, default=0.01, help='smooth weight')
    parser.add_argument('-loss_type', type=str, default='mse', help='type of loss function')
    parser.add_argument('-device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('-pin_memory', type=bool, default=False, help='pin_memory')
    parser.add_argument('-num_workers', type=int, default=0, help='num_workers')
    args = parser.parse_args()

    run(args)
