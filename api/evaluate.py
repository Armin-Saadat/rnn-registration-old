import os
import argparse
import torch

from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck
from path_definition import OUTPUT_DIR
from utils.losses import NCC, MSE, Grad


def run(args):
    dataloader = get_dataloader(args.batch_size, args.shuffle, args.pin_memory, args.num_workers)

    snapshot_path = os.path.join(OUTPUT_DIR, )
        snapshot = torch.load(snapshot_path, map_location='cpu')
        model = MODELS[snapshot['model_args'].type](snapshot['model_args'])
        model.load_state_dict(snapshot['model_state_dict'])
        loss_module = LOSSES[snapshot['loss_args'].type](snapshot['loss_args'])
        loss_module.load_state_dict(snapshot['loss_state_dict'])
        optimizer = OPTIMIZERS[snapshot['optimizer_args'].type](model.parameters(), snapshot['optimizer_args'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        return (
        model, loss_module, optimizer, snapshot['optimizer_args'], snapshot['epoch'], snapshot['train_reporter'],
        snapshot['valid_reporter'])


    model = Bottleneck(dataloader.dataset.image_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sim_loss = NCC().loss if args.loss == 'ncc' else MSE().loss
    smooth_loss = Grad('l2').loss
    seg_loss = MSE().loss

    trainer = Trainer(args, model, dataloader, optimizer, sim_loss, smooth_loss, seg_loss)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-snapshot', type=str, help='path to the saved snapshot.')
    parser.add_argument('-device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('-pin_memory', type=bool, default=False, help='pin_memory')
    parser.add_argument('-num_workers', type=int, default=0, help='num_workers')
    args = parser.parse_args()

    if args.snapshot is None:
        raise Exception('Please specify the path to your snapshot.')

    run(args)
