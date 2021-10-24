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



    def __visualize(self, images, labels, slice_idx):
        image = images[slice_idx].detach().cpu().numpy()
        label = labels[slice_idx].detach().cpu().numpy()
        pics = [image, label]
        titles = ['image', 'label']
        ne.plot.slices(pics, titles=titles, cmaps=['gray'], do_colorbars=True,
                       imshow_args=[{'origin': 'lower'}]);
