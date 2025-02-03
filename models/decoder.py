from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import ASPP


class Decoder(nn.Module):
    def __init__(self, in_channels, input_size, num_classes=1, atrous_rates: Sequence[int] = (12, 24, 36), d=256):
        super().__init__()
        self.input_size = input_size
        self.aspp = ASPP(in_channels, atrous_rates, d)
        self.scoring_layer = nn.Conv2d(d, num_classes, 1)

    def forward(self, x):
        x = self.aspp(x)
        x = self.scoring_layer(x)
        return F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, 
            groups=in_channels, bias=use_bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class Upsample(nn.Module):
    def __init__(self, out_channels):
        super(Upsample, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.blend_conv = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels, use_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_l, feat_s):
        feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        feat_arm = self.bottleneck(feat_l)
        return self.blend_conv(feat_up + feat_arm)


if __name__ == "__main__":
    batch_size = 1
    channels = 64
    H, W = 32, 32

    feat_l = torch.randn(batch_size, channels * 2, H, W)
    feat_s = torch.randn(batch_size, channels, H // 2, W // 2)
    up = Upsample(channels)
    out = up(feat_l, feat_s)
    print("Output shape: ", out.shape)
