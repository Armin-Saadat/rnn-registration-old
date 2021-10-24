import os
import argparse
import torch

from utils.dataloader import get_dataloader
from models.bottleneck import Bottleneck
from path_definition import OUTPUT_DIR



def run(args):
    dataloader = get_dataloader(args.batch_size, args.shuffle, args.pin_memory, args.num_workers)
    snapshot_path = os.path.join(OUTPUT_DIR, args.id, args.snapshot)
    snapshot = torch.load(snapshot_path, map_location='cpu')
    model = Bottleneck(dataloader.dataset.image_size)
    model.load_state_dict(snapshot['model_state_dict'])
    model = model.to(args.device)

    evaluator = Evaluator(args, model, dataloader)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('-id', type=str, help='id of your experiment.')
    parser.add_argument('-snapshot', type=str, help='name of the saved snapshot.')
    parser.add_argument('-device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('-pin_memory', type=bool, default=False, help='pin_memory')
    parser.add_argument('-num_workers', type=int, default=0, help='num_workers')
    args = parser.parse_args()

    if args.id is None:
        raise Exception('Please specify an ID for your run.')

    if args.snapshot is None:
        raise Exception('Please specify the name of your snapshot.')

    run(args)
