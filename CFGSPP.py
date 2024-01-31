import torch
from torch import nn
import numpy as np

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class CAGP(nn.Module):
    def __init__(self, channel, kernel, strides, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(CAGP, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(2, 1, kernel_size=3, padding=1, bias=False)
        self.AP = nn.AvgPool2d(kernel, strides, (kernel - 1) // 2, ceil_mode, count_include_pad, divisor_override)
        self.area = kernel * kernel

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        # y1 y2 -> channel information
        y = torch.cat([y1.squeeze(-1).transpose(-1, -2), y2.squeeze(-1).transpose(-1, -2)], 1)
        y = self.conv1(y)
        # downscale to get channel feature
        y = y.transpose(-1, -2).unsqueeze(-1)
        Temperature = torch.abs(y)
        Temperature = 10*torch.sigmoid(50*(Temperature-0.091))
        XoverT = x / Temperature.expand_as(x)
        x_exp = torch.exp(XoverT)
        denominator = self.AP(x_exp) * self.area
        weights = x_exp / denominator
        x = weights * x
        return x

class CFGSPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.GP5 = CAGP(kernel=5, channel=hidden_channels, strides=1)
        self.GP9 = CAGP(kernel=9, channel=hidden_channels, strides=1)
        self.GP13 = CAGP(kernel=13, channel=hidden_channels, strides=1)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.GP5(x)
        x2 = self.GP9(x1)
        x3 = self.GP13(x2)
        x = torch.cat([x, x1, x2, x3], 1)
        x = self.conv2(x)
        return x

