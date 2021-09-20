import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import os
import neurite as ne
from models.reg_rnn import REG_RNN
from utils import losses
import shutil
import pickle
import json
import random

# ______ ARGS ______ #
class Args():
    def __init__(self):
        self.model_id = '6'
        self.saving_base = '/HDD/kian/rnn-registration/saved-models/'
        self.model_dir = f'{self.saving_base}{self.model_id}/'
        self.train_mode = 'default (check 1e-4 lr) int_downsize=1'
#         self.supix_file_name = '/HDD/vxm-models/structured-data/filtered_labelMatched_lb85.pkl'
#         self.supix_top = None
#         self.supix_thr = None
        self.lr = 1e-4
        self.epochs = 400
        self.batch_size = 1
        self.smooth_weight = 0.01
        self.seg_weight = 0
        self.loss = 'mse'      
        self.load_model = None
        self.initial_epoch = 0  # to start from
#         self.int_steps = 7      # check 0
        self.int_downsize = 1  # check 0
        self.save_every = 50
        
        self.wait = 0


        assert self.loss == 'mse' or self.loss == 'ncc'

        for obj in os.listdir(self.saving_base):
            if obj == str(self.model_id):
                raise Exception(f'ID Error: model with ID {self.model_id} already exists in {self.model_dir}')
        os.mkdir(self.model_dir)


    def save(self):
        model_info =  f'\n___ Model Info: ID{self.model_id} ___\n'
        model_info += f'train_mode:     {self.train_mode}\n'
#         model_info += f'supix_top:      {self.supix_top} '
#         model_info += f'supix_thr:      {self.supix_thr} \n'
#         model_info += f'supix_file:     {self.supix_file_name}\n'
        model_info += f'___ model_dir:  {self.model_dir} ___\n\n'
        model_info += f'lr:             {self.lr}\n'
        model_info += f'epochs:         {self.epochs}\n'
        model_info += f'batch_size:     {self.batch_size}\n'
        model_info += f'smooth_weight:  {self.smooth_weight}\n'
        model_info += f'seg_weight:     {self.seg_weight}\n'
        model_info += f'loss:           {self.loss}\n'
        model_info += f'load_model:     {self.load_model}\n'
        model_info += f'initial epoch:  {self.initial_epoch}\n'
#         model_info += f'int_steps:      {self.int_steps}\n'
        model_info += f'int_downsize:   {self.int_downsize}\n'
        model_info += f'save_every:     {self.save_every}\n'
        print(model_info)

        with open(os.path.join(self.model_dir, f'model_{self.model_id}_info.txt'), 'w+') as info:
            info.write(model_info)
            info.close()

args = Args()
args.save()



# ______ Load and Prepare Data _______ #
server = True  # False means Colab

if server:
    with open('/HDD/vxm-models/structured-data/unfiltered_images.pkl', 'rb') as f:
        pre_images = pickle.load(f)

    with open('/HDD/vxm-models/structured-data/unfiltered_labels.pkl', 'rb') as f:
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
        
else:  
    with open('./data/images', 'rb') as f:
        images = pickle.load(f)

    with open('./data/labels', 'rb') as f:
        labels = pickle.load(f)
        
# Normalize        
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

dataloader = get_dataloader(images, labels, args.batch_size)


# _____ Model _____ #
model = REG_RNN(dataloader.dataset.image_size)
model.to('cuda')
_ = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.loss == 'ncc':
    sim_loss_func = losses.NCC().loss
elif args.loss == 'mse':
    sim_loss_func = losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

smooth_loss_func = losses.Grad('l2', loss_mult=args.int_downsize).loss
smooth_weight = args.smooth_weight

seg_loss_func = losses.MSE().loss
seg_weight = args.seg_weight

zero_phi = torch.zeros(39, 2, 256, 256).float().to('cuda')



# _____ Train _____ #
loss_history, all_metrics = [], []
for epoch in range(args.initial_epoch, args.epochs):
    time.sleep(args.wait)
    # save model checkpoint
    if epoch % args.save_every == 0:
        torch.save(model.state_dict(), os.path.join(args.model_dir, '%04d.pt' % epoch))
        with open(f'{args.model_dir}train_metrics.json', 'w+') as outfile:
            json.dump(all_metrics, outfile)

    epoch_loss = 0
    epoch_loss_sim = 0 
    epoch_loss_smooth = 0
    epoch_loss_seg = 0
    total_data_count = 0
    epoch_start_time = time.time()

    for d_idx, data in enumerate(dataloader):
        
        imgs, lbs = data
        bs = imgs.shape[0]

        # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
        imgs = imgs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')
        lbs = lbs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')

        # shape of moved_imgs/moved_labes: (seq_size - 1, bs, 1, W, H)
        # shape of flows: (seq_size - 1, bs, 2, W, H)
        moved_imgs, moved_lbs, flows = model(imgs, lbs)
        
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
        
    if epoch % 20 == 0:
        metrics = {
            'epoch': epoch,
            'epoch_loss': float(epoch_loss.item()),
            'epoch_loss_sim': epoch_loss_sim.item(),
            'epoch_loss_smooth': epoch_loss_smooth.item(),
            'epoch_loss_seg': epoch_loss_seg.item(),
            'total_data_count': total_data_count,
            'loss_history': (epoch_loss / total_data_count).item(),
        }
        all_metrics.append(metrics)
    
    # print epoch info
    loss_history.append(epoch_loss / total_data_count)
    msg = 'epoch %d/%d, ' % (epoch + 1, args.epochs)
    msg += 'loss= %.4e, ' % (epoch_loss / total_data_count)
    msg += 'sim_loss= %.4e, ' % (epoch_loss_sim / total_data_count)
    msg += 'smooth_loss= %.4e, ' % (epoch_loss_smooth / total_data_count)
    msg += 'seg_loss= %.4e, ' % (epoch_loss_seg / total_data_count)
    msg += 'time= %.4f, ' % (time.time() - epoch_start_time)
    print(msg, flush=True)
    
    
torch.save(model.state_dict(), os.path.join(args.model_dir, '%04d.pt' % args.epochs))