import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import os
from utils import losses
import shutil
import pickle
import json
import random

SERVER = 168
if SERVER == 168:
    base = '/home/khalili/kian-data/saved-models/'
elif SERVER == 166:
    base = '/HDD/kian/saved-models/DIR1/'

# ______ ARGS ______ #
class Args():
    def __init__(self, base):
        self.model_id = 'm9'
        self.saving_base = base
        self.model_dir = f'{self.saving_base}{self.model_id}/'
        self.train_mode = 'first try on new convlstm - 1 patient - after removing self.flow'
        self.lr = 1e-3
        self.epochs = 1200
        self.batch_size = 1
        self.smooth_weight = 0
        self.seg_weight = 0
        self.loss = 'mse'      
        self.load_model = None # '/HDD/kian/saved-models/DIR1/n7/0400.pt'
        self.initial_epoch = 0 # to start from
        self.save_every = 200
        self.wait = 0

        assert self.loss == 'mse' or self.loss == 'ncc'


    def save(self):
        model_info =  f'\n___ Model Info: ID{self.model_id} ___\n'
        model_info += f'train_mode:     {self.train_mode}\n'
        model_info += f'___ model_dir:  {self.model_dir} ___\n\n'
        model_info += f'lr:             {self.lr}\n'
        model_info += f'epochs:         {self.epochs}\n'
        model_info += f'batch_size:     {self.batch_size}\n'
        model_info += f'smooth_weight:  {self.smooth_weight}\n'
        model_info += f'seg_weight:     {self.seg_weight}\n'
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

args = Args(base)
args.save()


import torch
import torch.nn as nn
from models.unet_convlstm import Unet_ConvLSTM
from models.unet import Unet
# from models.conv_lstm import ConvLSTM
from models2.convlstm import ConvLSTM
from utils.spatial_transform import SpatialTransformer


class Unet_ConvLSTM(nn.Module):
    def __init__(self, image_size):
        super(Unet_ConvLSTM, self).__init__()
        self.image_size = image_size
        self.ndims = len(image_size)

        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.unet = Unet(inshape=image_size, infeats=2, nb_features=[enc_nf, dec_nf])

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % self.ndims)

        self.hidden_dim = 8
        self.flow = Conv(self.hidden_dim, kernel_size=3, padding=1)
        self.rnn = ConvLSTM(img_size=image_size, input_dim=self.unet.final_nf, hidden_dim=self.hidden_dim, kernel_size=(3, 3),
                            bidirectional=False, return_sequence=True, batch_first=False)
        self.spatial_transformer = SpatialTransformer(size=image_size)

    def forward(self, images, labels=None, convlstm=True):

        # shape of imgs/lbs: (seq_size, bs, 1, W, H)
        # shape of unet_out: (seq_size - 1, bs, 2, W, H)
        # shape of flows: (seq_size - 1, bs, 2, W, H)
        # shape of moved_images = (seq_size - 1, bs, 1, W, H)
        
        forward_sources, forward_targets = images[:-1], images[1:]
        src_trg_zip = zip(forward_sources, forward_targets)
        if convlstm:
            forward_unet_out = torch.cat([self.unet(torch.cat([src, trg], dim=1)).unsqueeze(0) for src, trg in src_trg_zip], dim=0)
            rnn_out, last_states, _ = self.rnn(forward_unet_out)
            forward_flows = self.flow(rnn_out[0].permute(1, 0, 2, 3, 4))
        else:
            forward_flows = torch.cat([self.flow(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0) for src, trg in src_trg_zip], dim=0)

        forward_moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(forward_sources, forward_flows[:])], dim=0)
        
        if labels is not None:
            forward_lbs_sources = labels[:-1]
            forward_moved_labels = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(forward_lbs_sources, forward_flows[:])], dim=0)
            
        backward_sources, backward_targets = images[1:], images[:-1]
        src_trg_zip = zip(backward_sources, backward_targets)
        if convlstm:
            backward_unet_out = torch.cat([self.unet(torch.cat([src, trg], dim=1)).unsqueeze(0) for src, trg in src_trg_zip], dim=0)
            rnn_out, last_states, _ = self.rnn(backward_unet_out)
            backward_flows = self.flow(rnn_out[0].permute(1, 0, 2, 3, 4))
        else:
            backward_flows = torch.cat([self.flow(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0) for src, trg in src_trg_zip], dim=0)

        backward_moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(backward_sources, backward_flows[:])], dim=0)
        
        if labels is not None:
            backward_lbs_sources = labels[1:]
            backward_moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(backward_lbs_sources, backward_flows[:])], dim=0)
            return forward_moved_images, forward_moved_labels, backward_moved_images, backward_moved_labels
        else:
            return forward_moved_images, backward_moved_images
            


# ______ Load and Prepare Data _______ #
if SERVER == 168:
    data_base = '/home/khalili/kian-data/'
elif SERVER == 166:
    data_base = '/HDD/vxm-models/structured-data/'
    
with open(f'{data_base}filtered_images.pkl', 'rb') as f:
    pre_images = pickle.load(f)

with open(f'{data_base}filtered_labels.pkl', 'rb') as f:
    pre_labels = pickle.load(f)

    
images, labels = [], []
for ind, img in pre_images.items():
    inp = torch.from_numpy(img)
    p_inp = torch.nn.functional.pad(inp, pad=(0, 0, 0, 0, 0, 40 - inp.shape[0]), mode='constant', value=0)
    images.append(p_inp)

for ind, img in pre_labels.items():
    inp = torch.from_numpy(img)
    p_inp = torch.nn.functional.pad(inp, pad=(0, 0, 0, 0, 0, 40 - inp.shape[0]), mode='constant', value=0)
    labels.append(p_inp)    
    

for i, img in enumerate(images):
    images[i] = (img/img.max()).float()

for i, lb in enumerate(labels):
    labels[i] = (lb/lb.max()).float()
    
    
    
    
# ____ Dataloader ____ #
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=workers)
    return dataloader

dataloader = get_dataloader(images[:1], labels[:1], args.batch_size)



# _____ Model _____ #
model = Unet_ConvLSTM(dataloader.dataset.image_size)
model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.5, patience=60, threshold=0.0001, verbose=True)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

if args.load_model:
    checkpoint = torch.load(args.load_model)
    print(f'\nloading model from {args.load_model}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('\nloaded\n')
    
model.train()

if args.loss == 'ncc':
    sim_loss_func = losses.NCC().loss
elif args.loss == 'mse':
    sim_loss_func = losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# smooth_loss_func = losses.Grad('l2').loss
# smooth_weight = args.smooth_weight
# seg_weight = args.seg_weight

seg_loss_func = losses.MSE().loss
zero_phi = torch.zeros(39, 2, 256, 256).float().to('cuda')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
# _____ Train _____ #
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
#         if d_idx % 4 == 0:
#             time.sleep(args.wait)
        
        # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
        # shape of moved_imgs/moved_labes: (seq_size - 1, bs, 1, W, H)
        # shape of flows: (seq_size - 1, bs, 2, W, H)
        
        imgs, lbs = data
        bs = imgs.shape[0]
        
        last_real = None
        for i, s in enumerate(imgs[0]):
            if s.max() < 1e-3:
                last_real = i - 1
                break
        imgs = imgs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')
        lbs = lbs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')
        
        loss, window = 0, 6
        for f, t in zip(np.arange(last_real + 1 - window), np.arange(window, last_real + 1)):
            fmoved, fmoved_labels, bmoved, bmoved_labels = model(imgs[f:t], lbs[f:t], convlstm=True)
            f_trgs, b_trgs = imgs[f:t][1:], imgs[f:t][:-1]
            loss += sim_loss_func(f_trgs, fmoved) + sim_loss_func(b_trgs, bmoved)
            if (f + 1) % 6 == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                loss = 0
        if loss != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
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
    msg += 'loss= %.4e, ' % (epc_loss)
    msg += 'time= %.4f, ' % (time.time() - epoch_start_time)
    print(msg, flush=True)
    
    
torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.model_dir, '%04d.pt' % (args.epochs)))
