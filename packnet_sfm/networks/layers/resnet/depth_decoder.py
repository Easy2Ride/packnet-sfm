# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import ConvBlock, Conv3x3, upsample


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, use_sub_pixel_convs=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        	
        self.use_sub_pixel_convs = use_sub_pixel_convs
        self.upscale_factor = 2

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
           	
            if self.use_sub_pixel_convs:	
                self.convs[("sub1", i)] = nn.Conv2d(num_ch_out, 64, 5, 1, 2)	
                self.convs[("sub2", i)] = nn.Conv2d(64, 32, 3, 1, 1)	
                self.convs[("sub_out", i)] = nn.Conv2d(32, num_ch_out*(self.upscale_factor**2), 3, 1, 1)	
                self.convs[("sub1", i)] = self.icnr_init(self.convs[("sub1", i)], self.upscale_factor)
                self.convs[("sub2", i)] = self.icnr_init(self.convs[("sub2", i)], self.upscale_factor)	
                self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
                
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        
        self.sigmoid = nn.Sigmoid()	
        self.relu    = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
    def icnr_init(self, conv, upsample_factor, init=nn.init.kaiming_normal_):
        """
        ICNR initialization for 2D/3D kernels adapted from Aitken et al.,2017 , "Checkerboard artifact free
        sub-pixel convolution".
        """
        tensor = conv.weight
        new_shape = [int(tensor.shape[0] / (upsample_factor ** 2))] + list(tensor.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, upsample_factor ** 2)
        transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        conv.weight.data.copy_(kernel)
   
        return conv

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if self.use_sub_pixel_convs:
                x = self.relu(self.convs[("sub1", i)](x))
                x = self.relu(self.convs[("sub2", i)](x))
                x = [self.pixel_shuffle(self.convs[("sub_out", i)](x))]
            else:
                x = [upsample(x)]
                
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                
                if self.num_output_channels==2 :	
                    self.outputs[("disp_uncertain", i)] = self.outputs[("disp", i)][:,1:,:,:]	
                    self.outputs[("disp", i)] = self.outputs[("disp", i)][:,:1,:,:]

        return self.outputs
