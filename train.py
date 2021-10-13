# __________________________________________________________________________________ IMPORTS
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import os
from utils import losses
import json
from models.unet_convlstm import Unet_ConvLSTM
from models.unet import Unet
from models2.convlstm import ConvLSTM
from utils.spatial_transform import SpatialTransformer


# __________________________________________________________________________ ARGS and CONFIG
class Args:
    def __init__(self):
        self.SERVER = 168
        self.model_id = 'slowUnet-2'
        self.model_description = ''
        self.all_patients = True
        self.one_patient = None
        self.use_rnn = True
        self.multi_windows = True
        self.use_filtered_dataset = True
        self.lr = 1e-4
        self.epochs = 2400
        self.batch_size = 1
        self.loss = 'mse'
        self.load_model = None   # '/HDD/kian/saved-models/DIR1/n7/0400.pt'
        self.initial_epoch = 0   # to start from
        self.save_every = 200
        self.cooldown_time = 0   # to decrease GPU temp
        self.lr_scheduler = 'ReduceLROnPlateau'

        if self.lr_scheduler == 'ReduceLROnPlateau':
            self.lr_scheduler_args = {'mode': 'min', 'factor': 0.75, 'patience': 20, 'threshold': 0.0001}
        elif self.lr_scheduler == 'ExponentialLR':
            self.lr_scheduler_args = {'gamma': 0.995}
        else:
            raise Exception("___LR Scheduler not implemented___")

        if self.SERVER == 168:
            self.saving_base = '/home/khalili/kian-data/saved-models/'
            self.data_base = '/home/khalili/kian-data/'
        elif self.SERVER == 166:
            self.saving_base = '/HDD/kian/saved-models/DIR1/'
            self.data_base = '/HDD/vxm-models/structured-data/'
        self.model_dir = f'{self.saving_base}{self.model_id}/'

    def save_info(self):
        model_info =  f'\n___ Model Info: ID {self.model_id} ___\n'
        model_info += f'model:          {self.model_description}\n'
        model_info += f'___ model_dir:  {self.model_dir} ___\n\n'
        model_info += f'lr:             {self.lr}\n'
        model_info += f'filtered imgs:   {self.use_filtered_dataset}\n'
        model_info += f'all patients:   {self.all_patients}\n'
        model_info += f'one patient:    {self.one_patient}\n'
        model_info += f'multi windows:  {self.multi_windows}\n'
        model_info += f'epochs:         {self.epochs}\n'
        model_info += f'batch_size:     {self.batch_size}\n'
        model_info += f'loss:           {self.loss}\n'
        model_info += f'load_model:     {self.load_model}\n'
        model_info += f'initial epoch:  {self.initial_epoch}\n'
        model_info += f'save_every:     {self.save_every}\n'
        print(model_info)

        for obj in os.listdir(self.saving_base):
            if obj == str(self.model_id):
                raise Exception(f'ID Error: model with ID {self.model_id} already exists in {self.model_dir}')
        os.mkdir(self.model_dir)

        with open(os.path.join(self.model_dir, f'model_{self.model_id}_info.txt'), 'w+') as info:
            info.write(model_info)
            info.close()


args = Args()
args.save_info()


# ______________________________________________________________________ Unet_ConvLSTM Class
class Unet_ConvLSTM(nn.Module):
    def __init__(self, image_size):
        super(Unet_ConvLSTM, self).__init__()
        self.image_size = image_size
        self.ndims = len(image_size)
        Conv = getattr(nn, 'Conv%dd' % self.ndims)

        # Unet
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.unet = Unet(inshape=image_size, infeats=2, nb_features=[enc_nf, dec_nf])
        self.unet_flow = Conv(in_channels=self.unet.final_nf, out_channels=2, kernel_size=3, padding=1)

        # RNN
        self.hidden_dim = 8
        self.flow = Conv(in_channels=self.hidden_dim, out_channels=4, kernel_size=3, padding=1)
        self.flow_2 = Conv(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        self.rnn = ConvLSTM(img_size=image_size, input_dim=self.unet.final_nf, hidden_dim=self.hidden_dim,
                            kernel_size=(3, 3), bidirectional=False, return_sequence=True, batch_first=False)

        self.spatial_transformer = SpatialTransformer(size=image_size)

    def forward(self, images, labels=None, use_rnn=True):

        # shape of imgs/lbs: (seq_size, bs, 1, W, H)
        # shape of unet_out: (seq_size - 1, bs, 2, W, H)
        # shape of flows: (seq_size - 1, bs, 2, W, H)
        # shape of moved_images = (seq_size - 1, bs, 1, W, H)

        # Forward: registering slice i to i + 1
        forward_sources, forward_targets = images[:-1], images[1:]
        src_trg_zip = zip(forward_sources, forward_targets)
        if use_rnn:
            forward_unet_out = torch.cat([self.unet(torch.cat([src, trg], dim=1)).unsqueeze(0)
                                          for src, trg in src_trg_zip], dim=0)
            rnn_out, last_states, _ = self.rnn(forward_unet_out)
            x = self.flow(rnn_out[0].squeeze(0))
            x = torch.nn.functional.leaky_relu(x)
            forward_flows = self.flow_2(x).unsqueeze(1)
        else:
            forward_flows = torch.cat([self.unet_flow(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0)
                                       for src, trg in src_trg_zip], dim=0)

        forward_moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0)
             for src, flow in zip(forward_sources, forward_flows[:])], dim=0)

        if labels is not None:
            forward_lbs_sources = labels[:-1]
            forward_moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0)
                 for src, flow in zip(forward_lbs_sources, forward_flows[:])], dim=0)

        # Backward: registering slice i to i - 1
        backward_sources, backward_targets = images[1:], images[:-1]
        src_trg_zip = zip(backward_sources, backward_targets)
        if use_rnn:
            backward_unet_out = torch.cat([self.unet(torch.cat([src, trg], dim=1)).unsqueeze(0)
                                           for src, trg in src_trg_zip], dim=0)
            rnn_out, last_states, _ = self.rnn(backward_unet_out)
            x = self.flow(rnn_out[0].squeeze(0))
            x = torch.nn.functional.leaky_relu(x)
            backward_flows = self.flow_2(x).unsqueeze(1)
        else:
            backward_flows = torch.cat([self.unet_flow(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0)
                                        for src, trg in src_trg_zip], dim=0)

        backward_moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0)
             for src, flow in zip(backward_sources, backward_flows[:])], dim=0)

        if labels is not None:
            backward_lbs_sources = labels[1:]
            backward_moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0)
                 for src, flow in zip(backward_lbs_sources, backward_flows[:])], dim=0)
            return forward_moved_images, forward_moved_labels, backward_moved_images, backward_moved_labels
        else:
            return forward_moved_images, backward_moved_images


# _____________________________________________________________________________ READING DATA
with open(f'{args.data_base}{"filtered" if args.use_filtered_dataset else "unfiltered"}_images.pkl', 'rb') as f:
    pre_images = pickle.load(f)
with open(f'{args.data_base}{"filtered" if args.use_filtered_dataset else "unfiltered"}_labels.pkl', 'rb') as f:
    pre_labels = pickle.load(f)

images, labels = [], []
for i, img in pre_images.items():
    images.append(torch.from_numpy(img))

for i, lb in pre_labels.items():
    labels.append(torch.from_numpy(lb))

# Normalize
for i, img in enumerate(images):
    images[i] = (img/img.max()).float()
for i, lb in enumerate(labels):
    labels[i] = (lb/lb.max()).float()


# __________________________________________________________________________________ DATA LOADER
class Dataset_(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.image_size = images[0].shape[1:]
        self.ndims = len(self.image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        outputs = [self.images[index], self.labels[index]]
        return tuple(outputs)


def get_dataloader(images, labels, batch_size, shuffle=False, pin_memory=False, workers=0):
    dataset = Dataset_(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            pin_memory=pin_memory, num_workers=workers)
    return dataloader


if args.all_patients:
    dataloader = get_dataloader(images, labels, args.batch_size)
elif args.one_patient:
    dataloader = get_dataloader([images[args.one_patient]], [labels[args.one_patient]], args.batch_size)
else:
    raise Exception(f"Ambiguity in Data set: all_patients: {args.all_patients}, one_patient: {args.one_patient}")



# __________________________________________________________________________ MODEL and OPTIMIZER
def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']

model = Unet_ConvLSTM(dataloader.dataset.image_size)
model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.lr_scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, **args.lr_scheduler_args, verbose=True)
elif args.lr_scheduler == 'ExponentialLR':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(**args.lr_scheduler_args)

if args.load_model is not None:
    checkpoint = torch.load(args.load_model)
    print(f'\n____ Loading model from {args.load_model} ____')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('\n_________ Model Loaded Successfully____\n')

model.train()

if args.loss == 'ncc':
    sim_loss_func = losses.NCC().loss
elif args.loss == 'mse':
    sim_loss_func = losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.loss)


# _____________________________________________________________________________________ TRAINING
loss_history, all_metrics = [], []
for epoch in range(args.initial_epoch, args.epochs):

    if (epoch + 1) % args.save_every == 0:
        print(f'------- lr: {get_lr(optimizer)} --------')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.model_dir, '%04d.pt' % (epoch + 1)))
        with open(f'{args.model_dir}train_metrics.json', 'w+') as outfile:
            json.dump(all_metrics, outfile)

    epoch_loss = 0
    total_data_count = 0
    epoch_start_time = time.time()

    for d_idx, data in enumerate(dataloader):

        # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
        # shape of moved_imgs/moved_labes: (seq_size - 1, bs, 1, W, H)
        # shape of flows: (seq_size - 1, bs, 2, W, H)
        imgs, lbs = data
        bs, num_layers = imgs.shape[0], imgs.shape[1]
        imgs = imgs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')
        lbs = lbs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')

        loss, win_count, window, step = 0, 0, 4, 2
        if args.multi_windows:
            if step == 0:
                data_ft = zip(np.arange(0, num_layers - window), np.arange(window, num_layers))
            else:
                data_ft = zip(np.arange(0, num_layers - window, step), np.arange(window, num_layers, step))
        else:
            data_ft = zip([0], [num_layers])

        for f, t in data_ft:
            fmoved, fmoved_labels, bmoved, bmoved_labels = model(imgs[f:t], lbs[f:t], use_rnn=args.use_rnn)
            f_trgs, b_trgs = imgs[f:t][1:], imgs[f:t][:-1]
            loss += sim_loss_func(f_trgs, fmoved) + sim_loss_func(b_trgs, bmoved)
            win_count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += (loss / win_count)
        total_data_count += 1

    epc_loss = float(epoch_loss) / total_data_count
    scheduler.step(epc_loss)

    if epoch % 20 == 0:
        metrics = {
            'epoch': epoch,
            'epoch_loss': epc_loss,
            'total_data_count': total_data_count,
            'current_lr': get_lr(optimizer),
        }
        all_metrics.append(metrics)

    # print epoch info
    loss_history.append(epc_loss)
    msg = 'epoch %d/%d, ' % (epoch + 1, args.epochs)
    msg += 'loss= %.4e, ' % epc_loss
    msg += 'time= %.4f, ' % (time.time() - epoch_start_time)
    print(msg, flush=True)

# Final save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.model_dir, '%04d.pt' % args.epochs))
