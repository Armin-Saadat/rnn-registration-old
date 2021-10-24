import time
import torch


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

        loss_history = []
        zero_phi = torch.zeros(39, 2, 256, 256).float().to(self.args.device)

        for epoch in range(self.args.epochs):

            epoch_loss = 0
            epoch_sim_loss = 0
            epoch_smooth_loss = 0
            epoch_seg_loss = 0
            total_data_count = 0
            epoch_start_time = time.time()

            for images, labels in self.dataloader:
                bs = images.shape[0]
                # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
                images = images.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)
                labels = labels.permute(1, 0, 2, 3).unsqueeze(2).to(self.args.device)

                # shape of moved_imgs/moved_lbs: (seq_size - 1, bs, 1, W, H)
                # shape of flow: (seq_size - 1, bs, 2, W, H)
                moved_imgs, moved_lbs, flow = self.model(images, labels)

                # calculate loss
                loss = 0
                for i in range(bs):
                    sim_loss = sim_loss_func(imgs[1:, i], moved_imgs[:, i])
                    smooth_loss = smooth_loss_func(zero_phi, flows[:, i])
                    seg_loss = seg_loss_func(lbs[1:, i], moved_lbs[:, i])
                    loss = sim_loss + (smooth_weight * smooth_loss) + (seg_weight * seg_loss)

                    # backpropagate and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update epoch loss
                    epoch_loss += loss * bs
                    epoch_loss_sim += sim_loss * bs
                    epoch_loss_smooth += smooth_loss * bs
                    epoch_loss_seg += seg_loss * bs
                    total_data_count += bs

                    # print epoch info
                loss_history.append(epoch_loss / total_data_count)
                msg = 'epoch %d/%d, ' % (epoch + 1, args.epochs)
                msg += 'loss= %.4e, ' % (epoch_loss / total_data_count)
                msg += 'sim_loss= %.4e, ' % (epoch_loss_sim / total_data_count)
                msg += 'smooth_loss= %.4e, ' % (epoch_loss_smooth / total_data_count)
                msg += 'seg_loss= %.4e, ' % (epoch_loss_seg / total_data_count)
                msg += 'time= %.4f, ' % (time.time() - epoch_start_time)
                print(msg, flush=True)
