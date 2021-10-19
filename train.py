# __________________________________________________________________________________ IMPORTS
import matplotlib.pyplot as plt
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
from models.unet import Unet
from models2.convlstm import ConvLSTM
from utils.spatial_transform import SpatialTransformer


# __________________________________________________________________________ ARGS and CONFIG
class Args:
    def __init__(self):
        """
        all_patients: bool, using all patients data in training.
        one_patient: int, index of one patient that model is going to use for training (0 or False if
                     using more than one patient.
        p_from, p_to: int, range of patients indexes to load as training data
                      (all_patients and one_patient both must be False).
        data_set_slice_step: int, number of slices to exclude from data set between each two slices that are taken.
        use_rnn: bool, if false, model will be pure Unet.
        depth: int (1 or 2), if 2, registering slice i to i + 2 additional to slice i, i + 1.
        multi_windows: bool, whether to split each 3D image into chunks of slices or not.
        use_filtered_dataset: bool, if true, all images in the dataset will have at least
                             one organ present in all slices.
        lr: float, learning rate to begin with.
        load_model: string, path of pre-trained model.
        initial_epoch: int, if using pre_trained model, epoch counter will begin with this number.
        save_every: int, number of epochs between two checkpoint savings.
        cooldown_time: int, number of seconds to wait between two epochs, waiting for the GPU temp to come down
                       (server 168 doesn't need this).
        """
        self.SERVER = 166
        self.model_id = 'rP1-fl-d'
        self.model_description = 'rnn.P1.w6-3. Hossein flow loss'
        self.all_patients = False
        self.one_patient = 1
        if not self.one_patient and not self.all_patients:
            self.p_from = 0
            self.p_to = 5
        self.dataset_slice_step = 0
        self.use_rnn = True
        self.depth = 2
        self.multi_windows = True
        self.use_filtered_dataset = True
        self.lr = 1e-4
        self.epochs = 4000
        self.batch_size = 1
        self.loss = 'mse'
        self.bidir_loss_weight = 1
        self.load_model = None   # '/home/khalili/kian-data/saved-models/x1/2500.pt'
        self.initial_epoch = 0   # to start from
        self.save_every = 500
        self.cooldown_time = 2   # to decrease GPU temp
        self.lr_scheduler = 'ReduceLROnPlateau'

        # if rnn
        self.rnn_hidden_dim = 64
        self.rnn_mid_flow_size = 32

        # Forcing multi_windows to be false while training a pure Unet
        if not self.use_rnn:
            self.multi_windows = False
        # if multi windows
        self.window = 6
        self.step = 3

        if self.lr_scheduler == 'ReduceLROnPlateau':
            self.lr_scheduler_args = {'mode': 'min', 'factor': 0.75, 'patience': 30, 'threshold': 0.0001}
        elif self.lr_scheduler == 'ExponentialLR':
            self.lr_scheduler_args = {'gamma': 0.995}
        else:
            raise Exception("___LR Scheduler not implemented___")

        if self.SERVER == 168:
            self.saving_base = '/home/khalili/kian-data/saved-models/'
            self.data_base = '/home/khalili/kian-data/'
        elif self.SERVER == 166:
            self.saving_base = '/HDD/kian/saved-models/cDir/'
            self.data_base = '/HDD/vxm-models/structured-data/'
        self.model_dir = f'{self.saving_base}{self.model_id}/'

    def save_info(self):
        model_info = f'\n___ Model Info: ID {self.model_id} ___\n'
        model_info += f'model:          {self.model_description}\n'
        model_info += f'___ model_dir:  {self.model_dir} ___\n\n'
        model_info += f'lr:             {self.lr}\n'
        model_info += f'use rnn:        {self.use_rnn}\n'
        model_info += f'depth:          {self.depth}\n'
        if self.use_rnn:
            model_info += f'rnn hdim:       {self.rnn_hidden_dim}\n'
            model_info += f'rnnMFS:         {self.rnn_mid_flow_size}\n'
        model_info += f'filtered imgs:  {self.use_filtered_dataset}\n'
        model_info += f'all patients:   {self.all_patients}\n'
        model_info += f'one patient:    {self.one_patient}\n'
        model_info += f'multi windows:  {self.multi_windows}\n'
        if self.multi_windows:
            model_info += f'window:         {self.window}\n'
            model_info += f'step:           {self.step}\n'
        model_info += f'epochs:         {self.epochs}\n'
        model_info += f'batch_size:     {self.batch_size}\n'
        model_info += f'loss:           {self.loss}\n'
        model_info += f'bidir weight:   {self.bidir_loss_weight}\n'
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
class Unet_RNN(nn.Module):
    def __init__(self, image_size, args):
        super(Unet_RNN, self).__init__()
        self.image_size = image_size
        self.ndims = len(image_size)
        Conv = getattr(nn, 'Conv%dd' % self.ndims)

        # Unet
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.unet = Unet(inshape=image_size, infeats=2, nb_features=[enc_nf, dec_nf])
        self.unet_downsize = Conv(in_channels=self.unet.final_nf, out_channels=2, kernel_size=3, padding=1)

        # RNN
        self.depth = args.depth
        self.hidden_dim = args.rnn_hidden_dim
        self.downsize_conv_1 = Conv(in_channels=self.hidden_dim, out_channels=args.rnn_mid_flow_size, kernel_size=3, padding=1)
        self.downsize_conv_2 = Conv(in_channels=args.rnn_mid_flow_size, out_channels=2, kernel_size=3, padding=1)
        self.rnn = ConvLSTM(img_size=image_size, input_dim=self.unet.final_nf, hidden_dim=self.hidden_dim,
                            kernel_size=(3, 3), bidirectional=False, return_sequence=True, batch_first=False)

        self.spatial_transformer = SpatialTransformer(size=image_size)

    def forward(self, images, labels=None, use_rnn=True):

        # shape of imgs/lbs: (seq_size, bs, 1, W, H)
        # shape of unet_out: (seq_size - 1, bs, 2, W, H)
        # shape of flows: (seq_size - 1, bs, 2, W, H)
        # shape of moved_images = (seq_size - 1, bs, 1, W, H)

        # Forward: registering slice i to i + 1
        if self.depth == 2:
            sources_, targets_ = torch.cat((images[:-1], images[:-2]), dim=0), torch.cat((images[1:], images[2:]), dim=0)
        elif self.depth == 1:
            sources_, targets_ = images[:-1], images[1:]
        else:
            raise Exception("___depth should be either 1 or 2___")
        src_trg_zip = zip(sources_, targets_)
        if use_rnn:
            unet_out = torch.cat(
                [self.unet(torch.cat([src, trg], dim=1)).unsqueeze(0) for src, trg in src_trg_zip], dim=0)
            rnn_out, last_states, _ = self.rnn(unet_out)
            x = rnn_out[0].squeeze(0)
            x = self.downsize_conv_1(x)
            x = torch.nn.functional.leaky_relu(x)
            flows = self.downsize_conv_2(x).unsqueeze(1)
        else:
            flows = torch.cat(
                [self.unet_downsize(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0)
                 for src, trg in src_trg_zip], dim=0)

        moved_images_ = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(sources_, flows[:])], dim=0)

        # Backward: registering moved image back to its source slice
        src_trg_zip = zip(moved_images_, sources_)
        if use_rnn:
            unet_out = torch.cat(
                [self.unet(torch.cat([src, trg], dim=1)).unsqueeze(0) for src, trg in src_trg_zip], dim=0)
            rnn_out, last_states, _ = self.rnn(unet_out)
            x = rnn_out[0].squeeze(0)
            x = self.downsize_conv_1(x)
            x = torch.nn.functional.leaky_relu(x)
            backward_flows = self.downsize_conv_2(x).unsqueeze(1)
        else:
            backward_flows = torch.cat(
                [self.unet_downsize(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0)
                 for src, trg in src_trg_zip], dim=0)

        backward_moved_images_ = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow
             in zip(moved_images_, backward_flows[:])], dim=0)

        if labels is not None:
            # We use labels only for testing, not training.
            lbs_sources_ = labels[:-1]
            moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0)
                 for src, flow in zip(lbs_sources_, flows[:])], dim=0)
            return sources_, targets_, moved_images_, moved_labels, backward_moved_images_, flows, backward_flows
        else:
            return sources_, targets_, moved_images_, backward_moved_images_


# _____________________________________________________________________________ READING DATA
with open(f'{args.data_base}{"filtered" if args.use_filtered_dataset else "unfiltered"}_images.pkl', 'rb') as f:
    pre_images = pickle.load(f)
with open(f'{args.data_base}{"filtered" if args.use_filtered_dataset else "unfiltered"}_labels.pkl', 'rb') as f:
    pre_labels = pickle.load(f)

if args.dataset_slice_step is False:
    bridge = 1
else:
    bridge = args.dataset_slice_step + 1

images, labels = [], []
for pid, img in pre_images.items():
    image_slices = []
    for i, slc in enumerate(img):
        if i % bridge == 0:
            image_slices.append(slc / slc.max())  # Normalizing
    images.append(torch.from_numpy(np.stack(image_slices, axis=0)).float())

for pid, lbs in pre_labels.items():
    label_slices = []
    for i, slc in enumerate(lbs):
        if i % bridge == 0:
            label_slices.append(slc)
    labels.append(torch.from_numpy(np.stack(label_slices, axis=0)).float())


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
    dataloader_ = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             pin_memory=pin_memory, num_workers=workers)
    return dataloader_


if args.all_patients:
    dataloader = get_dataloader(images, labels, args.batch_size)
    print(f'___dataloader filled with all data___')
elif args.one_patient:
    dataloader = get_dataloader([images[args.one_patient]], [labels[args.one_patient]], args.batch_size)
    print(f'___dataloader filled with patient number {args.one_patient}___')
elif args.p_to:
    dataloader = get_dataloader(images[args.p_from:args.p_to], labels[args.p_from:args.p_to], args.batch_size)
    print(f"____dataloader filled with patients range: {args.p_from}: {args.p_to}____")
else:
    raise Exception(f"Ambiguity in Data set: all_patients: {args.all_patients}, one_patient: {args.one_patient}"
                    f" p_from: {args.p_from}, p_to: {args.p_to}")


# __________________________________________________________________________ MODEL and OPTIMIZER
def get_lr(optimizer_):
    for param_group in optimizer_.param_groups:
        return param_group['lr']


model = Unet_RNN(dataloader.dataset.image_size, args)
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
bidir_weight = args.bidir_loss_weight
loss_history, all_metrics = [], []
for epoch in range(args.initial_epoch, args.epochs):
    time.sleep(args.cooldown_time)
    
    # Save model checkpoint and training metrics
    if (epoch + 1) % args.save_every == 0:
        print(f'------- lr: {get_lr(optimizer)} --------')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.model_dir, '%04d.pt' % (epoch + 1)))
        with open(f'{args.model_dir}train_metrics.json', 'w+') as outfile:
            json.dump(all_metrics, outfile)

    epoch_loss = 0
    patient_count = 0
    epoch_start_time = time.time()

    for d_idx, data in enumerate(dataloader):

        # shape of imgs/lbs: (bs, seq_size, W, H) --> (seq_size, bs, 1, W, H)
        # shape of moved_imgs/moved_labels: (seq_size - 1, bs, 1, W, H)
        # shape of flows: (seq_size - 1, bs, 2, W, H)
        imgs, lbs = data
        bs, num_layers = imgs.shape[0], imgs.shape[1]
        imgs = imgs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')
        lbs = lbs.permute(1, 0, 2, 3).unsqueeze(2).to('cuda')

        loss, window_count, window, step = 0, 0, args.window, args.step
        if args.multi_windows:
            if step == 0:
                data_ft = zip(np.arange(0, num_layers - window), np.arange(window, num_layers))
            else:
                data_ft = zip(np.arange(0, num_layers - window, step), np.arange(window, num_layers, step))
        else:
            data_ft = zip([0], [num_layers])

        for from_, to in data_ft:
            sources, targets, moved_images, _, backward_moved_images, ff, bf = model(imgs[from_:to], lbs[from_:to], use_rnn=args.use_rnn)
            sim_loss, bidir_loss = sim_loss_func(targets, moved_images), sim_loss_func(sources, backward_moved_images)
#             loss = torch.abs(ff + bf).sum()
            loss = sim_loss + (bidir_weight * bidir_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            window_count += 1
            epoch_loss += loss

        patient_count += 1

    epoch_loss = float(epoch_loss) / window_count
    sim_loss = float(sim_loss) / window_count
    bidir_loss = float(bidir_loss) / window_count
    scheduler.step(epoch_loss)

    if epoch % 20 == 0:
        metrics = {
            'epoch': epoch,
            'epoch_loss': epoch_loss,
            'sim_loss': sim_loss,
            'bidir_loss': bidir_loss,
            'current_lr': get_lr(optimizer),
        }
        all_metrics.append(metrics)

    # print epoch info
    loss_history.append(epoch_loss)
    msg = 'epoch %d/%d, ' % (epoch + 1, args.epochs)
    msg += 'loss= %.4e, ' % epoch_loss
    msg += 'time= %.4f, ' % (time.time() - epoch_start_time)
    print(msg, flush=True)


# Saving final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.model_dir, '%04d.pt' % args.epochs))
