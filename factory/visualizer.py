import neurite as ne
import torch


class Visualizer:
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model.eval()
        self.dataloader = dataloader

    def visualize(self):
        for i, data in enumerate(self.dataloader):
            if i != self.args.patient_idx:
                continue
            imgs, lbs = data
            # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
            imgs = imgs.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)
            lbs = lbs.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)

            # shape of moved_imgs/moved_labes: (seq_size - 1, bs, 1, W, H)
            # shape of flows: (seq_size - 1, bs, 2, W, H)
            with torch.no_grad():
                moved_imgs, moved_lbs, flows = self.model(imgs, lbs)

                for s in eval(self.args.slices):
                    self.__visualize(imgs[s, 0, 0], moved_imgs[s, 0, 0], imgs[s + 1, 0, 0])
                    self.__visualize(lbs[s, 0, 0], moved_lbs[s, 0, 0], lbs[s + 1, 0, 0])

    @staticmethod
    def __visualize(moving, moved, fixed):
        moving = moving.detach().cpu().numpy()
        moved = moved.detach().cpu().numpy()
        fixed = fixed.detach().cpu().numpy()
        pics = [moving, moved, fixed]
        titles = ['moving', 'moved', 'fixed']
        ne.plot.slices(pics, titles=titles, cmaps=['gray'], do_colorbars=True,
                       imshow_args=[{'origin': 'lower'}]);
