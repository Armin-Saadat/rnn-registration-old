import argparse
import torch

from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck
from utils.losses import NCC, MSE, Grad
from factory.trainer import Trainer


def run(args):
    dataloader = get_dataloader(args.batch_size, args.shuffle, args.pin_memory, args.num_workers)
    model = Bottleneck(dataloader.dataset.image_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sim_loss = NCC().loss if args.loss == 'ncc' else MSE().loss
    smooth_loss = Grad('l2').loss
    seg_loss = MSE().loss

    trainer = Trainer(args, model, dataloader, optimizer, sim_loss, smooth_loss, seg_loss)


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
