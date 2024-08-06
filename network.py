from torch.distributions.normal import Normal
import torch as th
import sys
sys.path.append('../') # add relative path
import torch.nn.functional as F
from torch import nn as nn
import numpy as np

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [th.arange(0, s) for s in size]
        grids = th.meshgrid(vectors)
        grid = th.stack(grids)
        grid = th.unsqueeze(grid, 0)
        grid = grid.type(th.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


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


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class Unet_multimodal(nn.Module):

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 outfeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 init=True):

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],  # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

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
        prev_nf_s = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        self.encoder_semantic = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            convs_semantic = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                convs_semantic.append(ConvBlock(ndims, prev_nf_s, nf))
                prev_nf_s = nf
                prev_nf = 2 * nf
            self.encoder.append(convs)
            self.encoder_semantic.append(convs_semantic)
            encoder_nfs.append(prev_nf // 2)

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
            self.remaining.append(ConvBlock(ndims, prev_nf + (3 if num==0 else 0), nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf
        if outfeats is None:
            if ndims == 2:
                self.flow = nn.Conv2d(self.final_nf, 1, kernel_size=3, padding=1)
            else:
                self.flow = nn.Conv3d(self.final_nf, 3, kernel_size=3, padding=1)
        else:
            if ndims == 2:
                self.flow = nn.Conv2d(self.final_nf, outfeats, kernel_size=3, padding=1)
            else:
                self.flow = nn.Conv3d(self.final_nf, outfeats, kernel_size=3, padding=1)
        if init:
            self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
            self.flow.bias = nn.Parameter(th.zeros(self.flow.bias.shape))

    def forward(self, x, y, z):

        # encoder forward pass
        x_history = [x]
        for level, (convs, convs_semantic) in enumerate(zip(self.encoder, self.encoder_semantic)):
            for conv in convs:
                x = conv(x)
            for conv in convs_semantic:
                y = conv(y)
            x_history.append(x)
            x = th.cat([x, y], dim=1)
            x = self.pooling[level](x)
            y = self.pooling[level](y)
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
                x = self.upsampling[level](x)
                x = th.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        x = th.cat([x, z], dim=1)
        for conv in self.remaining:
            x = conv(x)
        x = self.flow(x)
        return x

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec