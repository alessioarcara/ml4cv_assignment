import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConcatFusionBlock(nn.Module):
    def __init__(self, high_res_ch: int, decoder_ch: int) -> None:
        super().__init__()
        # Project high-res features to decoder channels using 1x1 conv
        self.reduce_high_res = nn.Sequential(
            nn.Conv2d(
                high_res_ch,
                decoder_ch,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU(inplace=True),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                decoder_ch * 2,
                decoder_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_res_feat: Tensor, low_res_feat: Tensor) -> Tensor:
        high_res_proj = self.reduce_high_res(high_res_feat)

        if low_res_feat.shape[2:] != high_res_feat.shape[2:]:
            low_res_upsampled = F.interpolate(
                low_res_feat,
                size=high_res_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            low_res_upsampled = low_res_feat

        fused = torch.cat([high_res_proj, low_res_upsampled], dim=1)

        return self.fusion_conv(fused)
