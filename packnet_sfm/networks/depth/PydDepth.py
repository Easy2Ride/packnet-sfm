# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.nn as nn
from functools import partial

from packnet_sfm.networks.layers.pydnet.pydnet import Pydnet
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth

########################################################################################################################

class PydDepth(Pydnet):
    """
    Wrapper for Pydnet which also includes disparity scaling
    """
    def __init__(self, enc_version="mobile_pydnet", dec_version="mobile_pydnet", pretrained=False, **kwargs):
        super(PydDepth, self).__init__(enc_version=enc_version, dec_version=dec_version, pretrained=pretrained )
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
