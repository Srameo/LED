from ultraled.archs.norm_util import MultipleScaleNorm2d, ScaleNorm2d
from ultraled.utils.registry import ARCH_REGISTRY

import torch
from torch import nn

### Normalization
from torch.nn import Identity
from torch.nn import BatchNorm2d, InstanceNorm2d
from ultraled.archs.norm_util import LayerNorm2d

import torch
from torch import nn
from torch.nn import functional as F
import math

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        out1 = self.conv1(x)
        out11 = self.lrelu(out1)
        out2 = self.conv2(out11)
        out21 = out2 + y

        return out21

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt


@ARCH_REGISTRY.register()
class CUNetArch(nn.Module):
    def __init__(self, inchannels=3, outchannels=3, channels=32) -> None:
        super().__init__()
        
        self.conv0_1 = nn.Linear(in_features=300, out_features=256)
        self.conv0_2 = nn.Linear(in_features=256, out_features=128)
        
        self.conv1_1 = nn.Conv2d(inchannels, channels, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        
        self.conv3_1 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = ResidualBlock(channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = ResidualBlock(channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(channels * 8, channels * 16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(channels * 16, channels * 16, kernel_size=3, stride=1, padding=1)


        self.upv6 = nn.ConvTranspose2d(channels * 16, channels * 8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(channels * 16, channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = ResidualBlock(channels * 8)


        self.upv7 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(channels * 8, channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = ResidualBlock(channels * 4)

        self.upv8 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.conv9_1 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(channels, outchannels, kernel_size=1, stride=1)

    def _check_and_padding(self, x):

        _, _, h, w = x.size()
        stride = (2 ** (5 - 1))

        dh = -h % stride
        dw = -w % stride

        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w+left_pad, top_pad, h+top_pad)

        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor

    def _check_and_crop(self, x):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top:bottom, left:right]
        return x
    
    def ratio_map_encoding(self, y):
        sigma = 30
        r = torch.arange(0, 300).cuda()

        
        Hr, Wr = 128, 128
        r = r.view(1, 300, 1, 1).expand(-1, -1, Hr, Wr)
        y = F.interpolate(y, size=(Hr, Wr), mode='bilinear', align_corners=False)
        r = torch.exp(-((r - y) ** 2) / (2 * sigma * sigma)) / (math.sqrt(2 * math.pi) * sigma)
        r = torch.mul(r, 1 / y)

        return r


    def forward(self, x, y, if_train=True):
        r = self.ratio_map_encoding(y)

        if if_train:
            Hc4, Wc4 = int(x.size(2) / 4), int(x.size(3) / 4)
        else:
            Hc4, Wc4 = int(x.size(2) / 4) + 2, int(x.size(3) / 4)
        r = F.interpolate(r, size=(Hc4, Wc4), mode='bilinear', align_corners=False)
        batch_size, _, H, W = r.shape
        r = r.view(batch_size, -1, H * W).permute(0, 2, 1)  


        # MLP
        control = self.conv0_1(r)  
        control = self.conv0_2(control)  
        control = control.permute(0, 2, 1).view(batch_size, 128, Hc4, Wc4)

        x = self._check_and_padding(x)
        
        
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3_1 = conv3 + control
        conv3 = self.lrelu(self.conv3_2(conv3, conv3_1))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4, conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6, conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3_1], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7, conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        out = self._check_and_crop(conv10)
        return out



    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt