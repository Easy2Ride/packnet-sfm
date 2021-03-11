# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.nn as nn
from functools import partial

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth

########################################################################################################################

class DepthResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, enc_version=None, pretrained=True, **kwargs):
        super().__init__()
        assert enc_version is not None, "DispResNet needs a version"
       

        self.encoder = ResnetEncoder(enc_type=enc_version, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

########################################################################################################################
