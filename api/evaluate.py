import os
import argparse
import torch

from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck
from path_definition import OUTPUT_DIR
from utils.losses import NCC, MSE, Grad


def run(args):
    dataloader = get_dataloader(args.batch_size, args.shuffle, args.pin_memory, args.num_workers)
    model = Bottleneck(dataloader.dataset.image_size).to(args.device)
    snapshot_path = os.path.join(OUTPUT_DIR, args.snapshot)
    snapshot = torch.load(snapshot_path, map_location='cpu')
    model.load_state_dict(snapshot['model_state_dict'])


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
