# Copyright 2020 Zeeshan Khan Suri.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.losses.loss_base import LossBase


class PoseConsistencyLoss(LossBase):
    """
    Pose consistency loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pose1, pose2, pose3):
        """Calculates the pose consistency loss
        
        Inputs
        -------
        pose1, pose2, pose3: torch.Tensor [Bx4x4], such that
            pose1 + pose2 = pose3

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary

        Author: Zeeshan Khan Suri
        """
        # pose12_trans = pose1[:, :3, -1]+pose2[:, :3, -1]
        # pose12_trans = pose12_trans.norm(dim=-1)
        # pose3_trans = pose3[:, :3, -1].norm(dim=-1)
        # Calculate pose consistency loss
        # loss = (pose12_trans - pose3_trans).abs().mean()

        # Translation consistency loss
        loss = (pose1[:,:3,-1]+pose2[:,:3,-1] - pose3[:,:3,-1]).abs().mean()
        # Rotation consistency loss R2 * R1 = R3
        loss += (torch.bmm(pose2[:,:2,:2], pose1[:,:2,:2]).norm(dim=-1) - pose3[:,:2,:2].norm(dim=-1)).abs().mean()
        self.add_metric('pose_consistency_loss', loss)
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }
