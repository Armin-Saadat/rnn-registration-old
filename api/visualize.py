import os
import argparse
import torch

from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck
from path_definition import OUTPUT_DIR
from factory.visualizer import Visualizer


def run(args):
    dataloader = get_dataloader(1, False, False, 0)
    snapshot_path = os.path.join(OUTPUT_DIR, args.id, args.snapshot)
    snapshot = torch.load(snapshot_path, map_location='cpu')
    model = Bottleneck(dataloader.dataset.image_size)
    model.load_state_dict(snapshot['model_state_dict'])
    model = model.to(args.device)

    visualizer = Visualizer(args, model, dataloader)
    visualizer.visualize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-id', type=str, help='id of your experiment.')
    parser.add_argument('-snapshot', type=str, help='name of the saved snapshot.')
    parser.add_argument('-patient_idx', type=int, help='index of the data.')
    parser.add_argument('-device', type=str, default='cuda', help='cpu or cuda')
    args = parser.parse_args()

    if args.id is None:
        raise Exception('Please specify an ID for your run.')

    if args.snapshot is None:
        raise Exception('Please specify the name of your snapshot.')

    run(args)
