import torch
import torch.nn as nn

from models.unet import Unet
from utils.spatial_transform import SpatialTransformer


class Unet_LSTM(nn.Module):
    def __init__(self, image_size):
        super(Unet_LSTM, self).__init__()
        self.image_size = image_size
        self.ndims = len(image_size)

        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.unet = Unet(inshape=image_size, infeats=2, nb_features=[enc_nf, dec_nf])

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % self.ndims)
        self.flow = Conv(self.unet.final_nf, self.ndims, kernel_size=3, padding=1)

        # features: H * w * 2 (for x and y in flow)
        features_num = image_size[0] * image_size[1] * 2
        self.rnn = nn.LSTM(input_size=features_num, hidden_size=features_num, batch_first=False)
        self.spatial_transformer = SpatialTransformer(size=image_size)

    def forward(self, images, labels=None):

        # shape of imgs/lbs: (seq_len, bs, 1, W, H)
        # shape of unet_out: (seq_len - 1, bs, 2, W, H)
        unet_out = torch.cat(
            [self.flow(self.unet(torch.cat([src, trg], dim=1))).unsqueeze(0)
             for src, trg in zip(images[:-1], images[1:])], dim=0)

        # shape of rnn_out: (seq_len - 1, bs, H*W*2)
        seq, bs, C, W, H = unet_out.shape
        rnn_out, (h1, c1) = self.rnn(unet_out.view(seq, bs, -1))

        # shape of flows: (seq_len - 1, bs, 2, W, H)
        flows = rnn_out.view(seq, bs, C, W, H)

        # shape of moved_images = (seq_len - 1, bs, 1, W, H)
        moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(images[:-1], flows[:])], dim=0)

        if labels is not None:
            moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(labels[:-1], flows[:])], dim=0)
            return [moved_images, moved_labels, flows]
        else:
            return [moved_images, flows]
