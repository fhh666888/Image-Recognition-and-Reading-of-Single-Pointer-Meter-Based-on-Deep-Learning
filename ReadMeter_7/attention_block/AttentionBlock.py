# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2022/3/21 15:16
# @Author   : Yang Jiaxiong
# @File     : AttentionBlock.py

import torch.nn as nn


class Attention_block(nn.Module):  # attention 就代替了原来U2Net的Res部分，因此concat的时候res的那一路就用attention代替
    def __init__(self, C_g, C_l, C_int):  # g 是上层粗糙的信号  l是本层细致的信号  两者通道数相同， int是中间转换的维度
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(C_g, C_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(C_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(C_l, C_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(C_int)
        )

        self.psi = nn.Sequential(  # 读作pu sai
            nn.Conv2d(C_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 需要保证两者输入的通道数相同
        batch_size = x.size()[0]
        assert batch_size == g.size(0)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # 注意这里可以直接乘，因为pytorch将维度不同的矩阵点乘做预处理了。
