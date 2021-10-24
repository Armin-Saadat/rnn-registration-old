import torch
import torch.nn as nn
import numpy as np

from utils.spatial_transform import SpatialTransformer


class Bottleneck(nn.Module):
    def __init__(self, image_size):
        super(Bottleneck, self).__init__()
        self.image_size = image_size
        self.ndims = len(image_size)

        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.unet = Unet2(inshape=image_size, infeats=2, nb_features=[enc_nf, dec_nf])

        self.lstm = nn.LSTM(input_size=32 * 16 * 16, hidden_size=32 * 16 * 16, batch_first=False)

        Conv = getattr(nn, 'Conv%dd' % self.ndims)
        self.flow = Conv(self.unet.final_nf, self.ndims, kernel_size=3, padding=1)

        self.fc = nn.Linear(2 * 32 * 32, 2 * 256 * 256)
        self.spatial_transformer = SpatialTransformer(size=image_size)

    def forward(self, images, labels=None):
        # shape of imgs/lbs: (40, bs, 1, 256, 256)
        T, bs = images.shape[0] - 1, images.shape[1]
        assert bs == 1, "batch-size must be one."
        assert T == 39, "sequence must be consisted of 40 slices."

        # shape of encoder_out: (39, bs, 32, 16, 16)
        X, X_history = [], []
        for src, trg in zip(images[:-1], images[1:]):
            x, x_history = self.unet(torch.cat([src, trg], dim=1), 'encode')
            X.append(x.unsqueeze(0))
            X_history.append(x_history)
        encoder_out = torch.cat(X, dim=0)

        # shape of lstm_out: (39, bs, 32, 16, 16)
        lstm_out, (h_n, c_n) = self.lstm(encoder_out.view(39, bs, -1))
        lstm_out = lstm_out.view(39, bs, 32, 16, 16)

        # shape of decoder_out: (39, bs, 2, 256, 256)
        Y = [self.flow(self.unet(lstm_out[i], 'decode', X_history[i])).unsqueeze(0) for i in range(T)]
        decoder_out = torch.cat(Y, dim=0)
        print(decoder_out.shape)
        exit()

        # shape of moved_images = (39, bs, 1, 256, 256)
        moved_images = torch.cat(
            [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(images[:-1], flows[:])], dim=0)

        if labels is not None:
            moved_labels = torch.cat(
                [self.spatial_transformer(src, flow).unsqueeze(0) for src, flow in zip(labels[:-1], flows[:])], dim=0)
            return [moved_images, moved_labels, flows]
        else:
            return [moved_images, flows]

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


class Unet2(nn.Module):
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):

        super().__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            enc_nf = [16, 32, 32, 32]
            dec_nf = [32, 32, 32, 32, 32, 16, 16]
            nb_features = [enc_nf, dec_nf]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x, task, x_history_=None):

        # encoder forward pass
        if task == 'encode':
            x_history = [x]
            for level, convs in enumerate(self.encoder):
                for conv in convs:
                    x = conv(x)
                x_history.append(x)
                x = self.pooling[level](x)

            return x, x_history

        # decoder forward pass with upsampling and concatenation
        elif task == 'decode':
            x_history = x_history_
            assert x_history is not None, "x_history_ is None."
            for level, convs in enumerate(self.decoder):
                for conv in convs:
                    x = conv(x)
                if not self.half_res or level < (self.nb_levels - 2):
                    x = self.upsampling[level](x)
                    x = torch.cat([x, x_history.pop()], dim=1)

            # remaining convs at full resolution
            for conv in self.remaining:
                x = conv(x)

            return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)

        return out
