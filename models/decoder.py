from typing import Sequence, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import ASPP
from models.fapn import FAM, FSM


class Decoder(nn.Module):
    def __init__(
        self, 
        fpn_channels: List[int],
        input_size: Tuple[int, int],
        num_classes: int, 
        atrous_rates: Sequence[int] = (6, 12, 18), 
        d: int = 128 
    ):
        super().__init__()
        self.input_size = input_size
        self.aspp = ASPP(fpn_channels[-1], atrous_rates, d)

        self.upsample_blocks = nn.ModuleList()
        for lateral_channels in fpn_channels[:-1][::-1]:
            self.upsample_blocks.append(
                Upsample(low_res_channels=d, lateral_channels=lateral_channels, out_channels=d)
            )

        self.scoring_layer = nn.Conv2d(d, num_classes, 1)

    def forward(self, fpn_features):
        x = self.aspp(fpn_features[-1])

        for up_module, feat in zip(self.upsample_blocks, fpn_features[:-1][::-1]):
            x = up_module(feat, x)

        x = self.scoring_layer(x)
        return F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)


class DecoderWithFAM(nn.Module):
    def __init__(
        self, 
        fpn_channels: List[int],
        num_classes: int,
        input_size,
        atrous_rates: Sequence[int] = (6, 12, 18), 
        d: int = 128 
    ):
        super().__init__()
        self.input_size = input_size
        self.aspp = ASPP(fpn_channels[-1], atrous_rates, d)
        #self.fsm_block = FSM(fpn_channels[-1], d) 

        self.fam_blocks = nn.ModuleList([
            FAM(lateral_channels, d) for lateral_channels in reversed(fpn_channels[:-1])
        ])

        self.scoring_layer = nn.Conv2d(d, num_classes, 1)

    def forward(self, fpn_features):
        x = self.aspp(fpn_features[-1])
        #x = self.fsm_block(fpn_features[-1])

        # from low to high resolution
        for fam_block, feat in zip(self.fam_blocks, fpn_features[:-1][::-1]):
            x = fam_block(feat, x)
        
        prelogits = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        logits = self.scoring_layer(prelogits)
        return prelogits, logits


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
    def __init__(self, low_res_channels, lateral_channels, out_channels):
        super(Upsample, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(lateral_channels, low_res_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_res_channels),
            nn.ReLU(inplace=True)
        )
        
        self.blend_conv = nn.Sequential(
            DepthwiseSeparableConv(low_res_channels, out_channels, use_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_l, feat_s):
        feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        feat_arm = self.bottleneck(feat_l)
        merged = feat_up + feat_arm
        return self.blend_conv(merged)


if __name__ == "__main__":
    fpn_channels = [64, 128, 256, 512]
    input_size = (256, 256)
    num_classes = 21

    decoder = Decoder(fpn_channels, input_size, num_classes)

    batch_size = 4
    fpn_features = [
        torch.randn(batch_size, 64, 256, 256),
        torch.randn(batch_size, 128, 128, 128),
        torch.randn(batch_size, 256, 64, 64),
        torch.randn(batch_size, 512, 32, 32)
    ]

    output = decoder(fpn_features)
    expected_shape = (batch_size, num_classes, input_size[0], input_size[1])
    assert output.shape == expected_shape