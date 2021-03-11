# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
	
from .resnet_utils import *
from .resnet import *


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=32, width_per_group=4, replace_stride_with_dilation=None,
                 norm_layer=None, num_input_images=2):
                 
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


def resnet_multiimage_input(enc_type, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    
    blocks = load_resnet_block(enc_type) 
    if (enc_type== "resnet18" or enc_type== "resnet34" or enc_type == "resnet18_ssl" or enc_type == "resnet18_swsl"):
        block_type = BasicBlock
    else:
        block_type = Bottleneck
    
    other_params = "".join([s for s in enc_type.split('_') if "32x" in s])
    if other_params != "":
        groups = int(other_params.split("x")[0])
        base_width = int(other_params.split("x")[1][0])
    else:
        groups, base_width = (1, 64)
    model = ResNetMultiImageInput(block_type, blocks, groups=groups, width_per_group=base_width, num_input_images=num_input_images)
    
    if pretrained:
        loaded = torch.hub.load_state_dict_from_url(load_model_url(enc_type))
        loaded["conv1.weight"] = torch.cat(
            [loaded["conv1.weight"]] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, enc_type, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(enc_type, pretrained, num_input_images)
        else:
            self.encoder = load_resnet_model(enc_type)(pretrained)

        if (enc_type != "resnet18" and enc_type!= "resnet34" and enc_type != "resnet18_ssl" and enc_type != "resnet18_swsl"):
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
