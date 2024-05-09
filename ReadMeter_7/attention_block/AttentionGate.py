# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/3/12 18:12
# @Author   : Yang Jiaxiong
# @File     : AttentionGate.py

import torch.nn.functional as F
from torch import nn


class Attention_Gate(nn.Module):
    """
    注意力门限，x和g的通道数相同，但是x的尺寸要求是g的两倍（也就是x是上一个stage的，g是本stage的）
    使用1*1的卷积核可以降低参数数量
    """

    def __init__(self, x_channels, g_channels):
        super(Attention_Gate, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(x_channels)
        )
        self.theta = nn.Conv2d(x_channels, x_channels, kernel_size=2, stride=2, padding=0, bias=False)

        self.phi = nn.Conv2d(g_channels, g_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Sequential(
            nn.Conv2d(x_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # 断言输入两者通道数一样
        input_size = x.size()  # 512, 18, 18
        batch_size = input_size[0]
        assert batch_size == g.size(0)  # 512, 9, 9

        theta_x = self.theta(x)  # 处理x， 先将x尺寸减半   x变为g的尺寸  512, 9, 9

        phi_g = self.phi(g)  # 处理g 维持g的尺寸，只不过是进行卷积  512, 9, 9
        psi = self.relu(theta_x + phi_g)

        psi = self.psi(psi)  # 组合 激活 1 9 9
        psi_up = F.interpolate(psi, size=input_size[2:], mode='bilinear', align_corners=True)  # 恢复到之前的x维度 1 18 18
        W_y = self.W(psi_up * x)  # todo  这里最后这个操作是否有用呢？
        return W_y
