import torch
import torch.nn as nn

from models.conv_lstm import ConvLSTM
from models.unet import Unet
from utils.spatial_transform import SpatialTransformer


class REG_RNN(nn.Module):
    def __init__(self, image_size):
        super(REG_RNN, self).__init__()
        self.image_size = image_size
        self.ndims = len(image_size)

        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.unet = Unet(inshape=image_size, infeats=2, nb_features=[enc_nf, dec_nf]).to('cuda')

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % self.ndims)
        self.flow = Conv(self.unet.final_nf, self.ndims, kernel_size=3, padding=1).to('cuda')

        self.rnn = ConvLSTM(input_dim=2, hidden_dim=2, kernel_size=(3, 3), num_layers=1, batch_first=False).to('cuda')
        self.spatial_transformer = SpatialTransformer(size=image_size).to('cuda')

    def forward(self, images, labels=None):

        # shape of imgs/lbs: (seq_size, bs, 1, W, H)
        # shape of unet_out: (seq_size - 1, bs, ndims, W, H)
        unet_out = torch.cat(
            [self.flow(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0)
             for src, trg in zip(images[:-1], images[1:])], dim=0)

        # shape of flows: (seq_size - 1, bs, 2, W, H)
        outputs, last_states = self.rnn(unet_out)
        flows = outputs[0].permute(1, 0, 2, 3, 4)
        h, c = last_states[0]

        # shape of moved_images = (seq_size - 1, bs, 1, W, H)
        moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(images[:-1], flows[:])], dim=0)

        if labels is not None:
            moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(labels[:-1], flows[:])], dim=0)
            return [moved_images, moved_labels, flows]
        else:
            return [moved_images, flows]
