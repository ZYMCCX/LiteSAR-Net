import torch
from torch import nn
import numpy as np
from CFGSPP import get_activation, BaseConv

class SConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = [0, 0]
        pad[0] = (ksize[0] - 1) // 2
        pad[1] = (ksize[1] - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SConv(in_channels=in_channels, out_channels=out_channels, ksize=(3, 13), stride=(2, 2), act="silu")
        self.conv2 = SConv(in_channels=in_channels, out_channels=out_channels, ksize=(13, 3), stride=(2, 2), act="silu")
        self.convside = BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=2, act="silu")
        #self.eca = eca_block(out_channels)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        #x3 = torch.cat([x1, x2], 1)
        #x4 = self.eca(self.convside(x))
        x3 = self.convside(x)
        return x1 + x2 + x3

class toup(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = SConv(in_channels=in_channels, out_channels=out_channels, ksize=(3, 13), stride=(1, 1),
                           act="silu")
        self.conv2 = SConv(in_channels=in_channels, out_channels=out_channels, ksize=(13, 3), stride=(1, 1),
                           act="silu")
        self.convside = BaseConv(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=1, act="silu")
        #self.eca = eca_block(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        #x3 = torch.cat([x1, x2], 1)
        #x4 = self.eca(self.convside(x))
        x3 = self.convside(x)
        return x1 + x2 + x3

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x

class E2Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #print(in_channels, out_channels)

        self.downsample1 = down(in_channels, out_channels)
        self.downsample2 = down(out_channels, out_channels*2)
        self.CBAM = cbam_block(out_channels*2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample1 = toup(out_channels * 2, out_channels)
        self.upsample2 = toup(out_channels, 3)

    def forward(self, x):

        x1 = self.downsample1(x)
        #print("x1:",x1.shape)
        # 256 256 3  -> 128 128 24
        x2 = self.downsample2(x1)
        #print("x2:", x2.shape)
        # 128 128 24 -> 64  64  48
        x2 = self.CBAM(x2)

        x3 = self.upsample1(x2)

        #print("x3:", x3.shape)

        # 64  64  48 -> 64  64  24
        xup = self.upsample(x3)

        #print("xup:", xup.shape)
        # 64  64  48 -> 128 128 24

        x4 = x1+xup

        x5 = self.upsample2(x4)

        xup = self.upsample(x5)

        x6 = xup+x

        return x6