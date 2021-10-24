import os
import time
import matplotlib.pyplot as plt
import torch

from path_definition import OUTPUT_DIR


class Trainer:
    def __init__(self, args, model, dataloader, optimizer, sim_loss, smooth_loss, seg_loss):
        self.args = args
        self.model = model.train()
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.sim_loss = sim_loss
        self.smooth_loss = smooth_loss
        self.seg_loss = seg_loss

    def train(self):
        self.setup_training_dir()
        self.__train()
        self.save_snapshot()
        self.save_plot()

    def __train(self):

        self.loss_history = []
        zero_phi = torch.zeros(39, 2, 256, 256).float().to(self.args.device)

        for epoch in range(self.args.epochs):

            epoch_loss = 0
            epoch_sim_loss = 0
            epoch_smooth_loss = 0
            epoch_seg_loss = 0
            epoch_data_n = 0
            epoch_start_time = time.time()

            for imgs, lbs in self.dataloader:
                bs = imgs.shape[0]
                # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
                imgs = imgs.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)
                lbs = lbs.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)

                # shape of moved_imgs/moved_lbs: (seq_size - 1, bs, 1, W, H)
                # shape of flow: (seq_size - 1, bs, 2, W, H)
                moved_imgs, moved_lbs, flow = self.model(imgs, lbs)

                # calculate loss
                loss = 0
                for i in range(bs):
                    sim_loss = self.sim_loss(imgs[1:, i], moved_imgs[:, i])
                    smooth_loss = self.smooth_loss(zero_phi, flow[:, i])
                    seg_loss = self.seg_loss(lbs[1:, i], moved_lbs[:, i])
                    loss = (self.args.sim_w * sim_loss) + (self.args.smooth_w * smooth_loss) + (
                            self.args.seg_w * seg_loss)

                # backpropagate and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update epoch loss
                epoch_loss += loss * bs
                epoch_sim_loss += sim_loss * bs
                epoch_smooth_loss += smooth_loss * bs
                epoch_seg_loss += seg_loss * bs
                epoch_data_n += bs

            # print epoch info
            self.loss_history.append(epoch_loss / epoch_data_n)
            msg = 'epoch %d/%d, ' % (epoch + 1, self.args.epochs)
            msg += 'loss= %.4e, ' % (epoch_loss / epoch_data_n)
            msg += 'sim_loss= %.4e, ' % (epoch_sim_loss / epoch_data_n)
            msg += 'smooth_loss= %.4e, ' % (epoch_smooth_loss / epoch_data_n)
            msg += 'seg_loss= %.4e, ' % (epoch_seg_loss / epoch_data_n)
            msg += 'time= %.4f, ' % (time.time() - epoch_start_time)
            print(msg, flush=True)

    def setup_training_dir(self):
        os.makedirs(os.path.join(OUTPUT_DIR, self.args.id), exist_ok=False)
        os.makedirs(os.path.join(OUTPUT_DIR, self.args.id, 'visualization'), exist_ok=False)

    def save_snapshot(self):
        snapshot = {'model_state_dict': self.model.state_dict()}
        torch.save(snapshot, os.path.join(OUTPUT_DIR, self.args.id, '%03d.pt' % self.args.epochs))
        del snapshot

    def save_plot(self):
        plt.plot(self.loss_history)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(OUTPUT_DIR, self.args.id, 'loss.png'))
        plt.close()
