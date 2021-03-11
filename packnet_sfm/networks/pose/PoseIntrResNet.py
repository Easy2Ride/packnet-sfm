# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.pose_decoder import PoseDecoder
from packnet_sfm.networks.layers.resnet.intrinsics_network import IntrinsicsNetwork

########################################################################################################################

class PoseIntrResNet(nn.Module):
    """
    Pose network based on the ResNet architecture.

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
    def __init__(self, version=None, width=640, height=192, **kwargs):
        super().__init__()
        assert version is not None, "PoseResNet needs a version"
        
        enc_type = version.split("-")[0]      # First two characters are the number of layers
        pretrained = version.split("-")[1]  == 'pt'    # If the last characters are "pt", use ImageNet pretraining

        self.encoder = ResnetEncoder(enc_type=enc_type, pretrained=pretrained, num_input_images=2)
        self.resize_len = torch.tensor([[width,height]],device='cuda')
        self.intr_net = IntrinsicsNetwork(self.encoder.num_ch_enc, self.resize_len)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

    def forward(self, target_image, ref_imgs):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs = []
        intr = []
        for i, ref_img in enumerate(ref_imgs):
            inputs = torch.cat([target_image, ref_img], 1)
            axisangle, translation = self.decoder([self.encoder(inputs)])
            intr_K = self.intr_net([self.encoder(inputs)])
            outputs.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
            intr.append(intr_K)
        pose = torch.cat(outputs, 1)
        #print(pose.shape)

        return pose, intr_K

########################################################################################################################

