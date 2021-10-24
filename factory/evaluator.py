import os
import torch
import numpy as np

from path_definition import OUTPUT_DIR
from utils.losses import Dice


class Evaluator:
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model.eval()
        self.dataloader = dataloader
        self.dice_loss = Dice().loss

    def evaluate(self):
        dice_score = []
        for imgs, lbs in self.dataloader:
            bs = imgs.shape[0]
            # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
            imgs = imgs.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)
            lbs = lbs.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)

            # shape of moved_imgs/moved_labes: (seq_size - 1, bs, 1, W, H)
            # shape of flows: (seq_size - 1, bs, 2, W, H)
            with torch.no_grad():
                moved_imgs, moved_lbs, flows = self.model(imgs, lbs)

            for i in range(bs):
                zero = torch.zeros_like(lbs[1:, i]).to(self.args.device)
                one = torch.ones_like(lbs[1:, i]).to(self.args.device)
                fixed_lbs_ = torch.where(lbs[1:, i] > 0, one, zero)
                moved_lbs_ = torch.where(moved_lbs[:, i] > 0, one, zero)

                # exclude padding
                labeled = []
                for k in range(fixed_lbs_.shape[0]):
                    if fixed_lbs_[k].max() != 0.0 and moved_lbs_[k].max() != 0.0:
                        labeled.append(k)
                fixed_lbs_ = fixed_lbs_[labeled]
                moved_lbs_ = moved_lbs_[labeled]

                dice_score.append((-self.dice_loss(fixed_lbs_, moved_lbs_)).detach().cpu().numpy())

        print('mean: ' + str(np.mean(dice_score)) + ' , std: ' + str(np.std(dice_score)))
        f = open(os.path.join(OUTPUT_DIR, self.args.id, 'eval.txt'), "w")
        f.write('mean: ' + str(np.mean(dice_score)) + ' , std: ' + str(np.std(dice_score)))
        f.close()
