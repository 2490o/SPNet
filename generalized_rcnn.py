# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F


from Encoder import Encoder


import pdb

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self):
        super(GeneralizedRCNN, self).__init__()
        model = Encoder()
        self.backbone = model.encoder
        self.netup = torch.nn.Sequential(
                torch.nn.Conv2d(512, 30, 3, padding=1)
                )
        self.centroids = torch.nn.Parameter(torch.rand(30, 512))   #生成（24, 256)可训练学习的张量
        self.num_cluster = 30
        self.upfc = torch.nn.Linear(30*512, 512)

        self.transform = torch.nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=False),
            )

    def UP(self, scene):
        x = scene

        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)          #对x做正则化，除2范数
        soft_assign = self.netup(x)                #通道数变为24

        soft_assign = F.softmax(soft_assign, dim=1)  #通道注意力机制
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)
        #调整图的大小
        x_flatten = x.view(N, C, -1)

        centroid = self.centroids       #生成（24, 256)可训练学习的张量

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3) #对维度进行扩展
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)#在0处增加一个维度

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc(up).unsqueeze(2).unsqueeze(3).repeat(1,1,W,H)

        return up, centroid

    def forward(self, images):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        # images = to_image_list(images)
        src_features = self.backbone(images)

        up, centroid = self.UP(src_features[3])
        new_features = torch.cat((src_features[3], up), dim=1)
        new_features = self.transform(new_features)

        up_features = []
        for i in range(len(src_features)-1):
            up_features.append(src_features[i])
        up_features.append(new_features)
        features = up_features

        return features
