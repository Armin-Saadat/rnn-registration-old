import argparse
import torch

from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck
from utils.losses import NCC, MSE, Grad


def run(args):
    dataloader = get_dataloader(args.batch_size, args.shuffle, args.pin_memory, args.num_workers)
    model = Bottleneck(dataloader.dataset.image_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == 'ncc':
        sim_loss_func = NCC().loss
    elif args.loss == 'mse':
        sim_loss_func = MSE().loss
    else:
        raise ValueError('Image loss should be either "mse" or "ncc", but found "%s"' % args.loss_type)



    smooth_loss_func = Grad('l2').loss
    smooth_weight = args.smooth_w

    seg_loss_func = MSE().loss
    seg_weight = args.seg_w

    for images, labels in dataloader:
        images = images.permute(1, 0, 2, 3).unsqueeze(2).to(args.device)
        labels = labels.permute(1, 0, 2, 3).unsqueeze(2).to(args.device)
        model(images, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-lr', type=float, default=0.001, help='learning-rate')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-smooth_w', type=float, default=0.01, help='weight of smooth loss')
    parser.add_argument('-seg_w', type=float, default=0.01, help='weight of segmentation loss')
    parser.add_argument('-loss_type', type=str, default='mse', help='type of loss function')
    parser.add_argument('-device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('-pin_memory', type=bool, default=False, help='pin_memory')
    parser.add_argument('-num_workers', type=int, default=0, help='num_workers')
    args = parser.parse_args()

    run(args)
