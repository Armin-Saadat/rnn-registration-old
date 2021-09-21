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

        convs = []
        convs.append(nn.Conv2d(self.unet.final_nf, 8, kernel_size=32, padding=0))
        convs.append(nn.Conv2d(8, 4, kernel_size=32, padding=0))
        convs.append(nn.Conv2d(4, 2, kernel_size=32, padding=0))
        for i in range(4):
            convs.append(nn.Conv2d(2, 2, kernel_size=32, padding=0))
        convs.append(nn.Conv2d(2, 2, kernel_size=8, padding=0))
        self.convs = nn.ModuleList(convs)

        self.rnn = nn.LSTM(input_size=32 * 32 * 2, hidden_size=32 * 32 * 2, batch_first=False)

        self.fc = nn.Linear(2 * 32 * 32, 2 * 256 * 256)

        self.spatial_transformer = SpatialTransformer(size=image_size)

    def forward(self, images, labels=None):

        # shape of imgs/lbs: (40, bs, 1, 256, 256)
        # shape of unet_out: (39, bs, 16, 256, 256)
        unet_out = torch.cat(
            [self.unet(torch.cat([src, trg], dim=1)).unsqueeze(0)
             for src, trg in zip(images[:-1], images[1:])], dim=0)

        assert unet_out.shape[1] == 1, "batch-size must be one"

        # shape of convs_out: (39, 2, 32, 32)
        convs_out = self.convs(unet_out.squeeze(1)).unsqueeze(1)

        # shape of rnn_out: (39, bs, 2*32*32)
        rnn_out, (h1, c1) = self.rnn(convs_out.view(39, 1, -1))

        # shape of flows: (39, bs, 2, 256, 256)
        flows = self.fc(rnn_out.view(39, -1)).view(39, 1, 2, 256, 256)

        # shape of moved_images = (seq_len - 1, bs, 1, W, H)
        moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(images[:-1], flows[:])], dim=0)

        if labels is not None:
            moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(labels[:-1], flows[:])], dim=0)
            return [moved_images, moved_labels, flows]
        else:
            return [moved_images, flows]
