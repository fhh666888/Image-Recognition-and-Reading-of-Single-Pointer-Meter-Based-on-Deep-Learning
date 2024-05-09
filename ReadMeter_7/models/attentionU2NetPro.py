# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/12/23 17:51
# @Author   : Yang Jiaxiong
# @File     : attentionU2Net.py  内部的每一个stage每一层都增加attention gate 外部没有注意力机制

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_block import Attention_Gate, Attention_block
from models.net import REBNCONV


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(input=src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src


# 考虑到上层需要注意力机制，所以保留注意力
class AttU_Net7(nn.Module):  # stage1  将图片从3维扩展到32维，但是尺寸从288变为9，再从9变为288（保持通过前后照片尺寸不变）
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AttU_Net7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 最深的一层，两侧对称

        # 注意这里两个输入的张量尺寸一样，所以用的是Attention block。因为进行的是空洞卷积（平向传输）
        self.attention6 = Attention_block(mid_ch, mid_ch, mid_ch)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention5 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention4 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention3 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention2 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention1 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)  # 最后重整

    def forward(self, x):
        hx = x  # hx不断在变化（/2)
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)  # 32, 18, 18
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)  # 由于这里hx6没有过pool，所以hx6和hx7的维度相同！

        a6 = self.attention6(g=hx7, x=hx6)  # 32, 9, 9
        hx6d = self.rebnconv6d(torch.cat((hx6, a6), dim=1))  # 平行传递层 32, 9, 9

        a5 = self.attention5(g=hx6d, x=hx5)  # 32, 18, 18
        hx6dup = _upsample_like(hx6d, hx5)  # 32, 18, 18
        hx5d = self.rebnconv5d(torch.cat((a5, hx6dup), 1))  # 32, 18, 18

        a4 = self.attention4(g=hx5d, x=hx4)  # 32,36,36
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((a4, hx5dup), 1))

        a3 = self.attention3(g=hx4d, x=hx3)  # 32,72,72
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((a3, hx4dup), 1))

        a2 = self.attention2(g=hx3d, x=hx2)  # 32,144,144
        hx3dup = _upsample_like(hx3d, hx2)  # 32,144,144
        hx2d = self.rebnconv2d(torch.cat((a2, hx3dup), 1))  # 32,144,144

        a1 = self.attention1(g=hx2d, x=hx1)  # 32,288,288
        hx2dup = _upsample_like(hx2d, hx1)  # 32,288,288
        hx1d = self.rebnconv1d(torch.cat((a1, hx2dup), 1))

        return hx1d + hxin


class AttU_Net6(nn.Module):  # 128,144,144  相比上一个stage通道扩大四倍，尺寸减小一倍
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AttU_Net6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        # 注意这里两个输入的张量尺寸一样，所以用的是Attention block。因为进行的是空洞卷积（平向传输）
        self.attention5 = Attention_block(mid_ch, mid_ch, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention4 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention3 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention2 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention1 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):  # 64,144,144  注意这里拿到的尺寸就已经降低了，这是因为外层pooling导致的
        hx = x

        hxin = self.rebnconvin(hx)  # 128,144,144

        hx1 = self.rebnconv1(hxin)  # 32,144,144
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)  # 32,72,72
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)  # 32,36,36
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)  # 32,18,18
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        a5 = self.attention5(g=hx6, x=hx5)
        hx5d = self.rebnconv5d(torch.cat((a5, hx6), 1))  # 平行传递层

        a4 = self.attention4(g=hx5d, x=hx4)
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((a4, hx5dup), 1))

        a3 = self.attention3(g=hx4d, x=hx3)
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((a3, hx4dup), 1))

        a2 = self.attention2(g=hx3d, x=hx2)
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((a2, hx3dup), 1))

        a1 = self.attention1(g=hx2d, x=hx1)  # 32,144,144
        hx2dup = _upsample_like(hx2d, hx1)  # 32,144,144
        hx1d = self.rebnconv1d(torch.cat((a1, hx2dup), 1))  # 128,144,144

        return hx1d + hxin


class AttU_Net5(nn.Module):  # 256,72,72
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AttU_Net5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        # 注意这里两个输入的张量尺寸一样，所以用的是Attention block。因为进行的是空洞卷积（平向传输）
        self.attention4 = Attention_block(mid_ch, mid_ch, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention3 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention2 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention1 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):  # 128,72,72
        hx = x

        hxin = self.rebnconvin(hx)  # 256,72,72

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        a4 = self.attention4(g=hx5, x=hx4)
        hx4d = self.rebnconv4d(torch.cat((a4, hx5), 1))  # 平行传递

        a3 = self.attention3(g=hx4d, x=hx3)
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((a3, hx4dup), 1))

        a2 = self.attention2(g=hx3d, x=hx2)
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((a2, hx3dup), 1))

        a1 = self.attention1(g=hx2d, x=hx1)
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((a1, hx2dup), 1))  # 256,72,72

        return hx1d + hxin


class AttU_Net4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AttU_Net4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        # # 注意这里两个输入的张量尺寸一样，所以用的是Attention block。因为进行的是空洞卷积（平向传输）
        self.attention3 = Attention_block(mid_ch, mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention2 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.attention1 = Attention_Gate(mid_ch, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):  # 256,36,36
        hx = x

        hxin = self.rebnconvin(hx)  # 512,36,36

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        a3 = self.attention3(g=hx4, x=hx3)
        hx3d = self.rebnconv3d(torch.cat((a3, hx4), 1))  # 平行传递层

        a2 = self.attention2(g=hx3d, x=hx2)
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((a2, hx3dup), 1))

        a1 = self.attention1(g=hx2d, x=hx1)
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((a1, hx2dup), 1))  # 512,36,36

        return hx1d + hxin


class AttU_Net4F(nn.Module):  # 不再使用升降采样改变尺寸和维度
    # 是一个退化类型——dirated convolution  注意这里也就不再进行注意力了，因为这里都是抽象语义，希望能够保留，注意力再次进行筛选的代价较高
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(AttU_Net4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):  # 512,18,18
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)  # 256,18,18
        hx2 = self.rebnconv2(hx1)  # 256,18,18
        hx3 = self.rebnconv3(hx2)  # 256,18,18

        hx4 = self.rebnconv4(hx3)  # 256,18,18

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))  # 256,18,18
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))  # 256,18,18
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))  # 512,18,18

        return hx1d + hxin


# AG Only U2Net Pro
class AttU2NetPro(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AttU2NetPro, self).__init__()

        self.stage1 = AttU_Net7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = AttU_Net6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = AttU_Net5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = AttU_Net4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = AttU_Net4F(512, 256, 512)  # 没有升降采样
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = AttU_Net4F(512, 256, 512)  # 最下面一层

        # decoder
        self.stage5d = AttU_Net4F(1024, 256, 512)
        self.stage4d = AttU_Net4(1024, 128, 256)
        self.stage3d = AttU_Net5(512, 64, 128)
        self.stage2d = AttU_Net6(256, 32, 64)
        self.stage1d = AttU_Net7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1  这个过程就是不断增加通道数，减少图片大小的过程
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)

