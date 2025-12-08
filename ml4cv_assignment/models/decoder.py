from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ml4cv_assignment.models.layers import ASPP, FAM, ConcatFusionBlock


class BaseDecoder(nn.Module, ABC):
    def __init__(
        self,
        fpn_channels: List[int],
        num_classes: int,
        decoder_channels: int,
        input_size: Tuple[int, int],
        output_stride: int,
        use_aspp: bool = True,
        atrous_rates: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.input_size = input_size

        # The last layer of the backbone (lower resolution / higher semantic) enters into the decoder
        high_level_ch = fpn_channels[-1]

        self.aspp: nn.Module
        if use_aspp:
            if atrous_rates is None:
                atrous_rates = self._compute_rates(input_size[0], output_stride)
            self.aspp = ASPP(high_level_ch, atrous_rates, decoder_channels)
        else:
            self.aspp = nn.Sequential(
                nn.Conv2d(high_level_ch, decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
            )

        self.fusion_blocks = nn.ModuleList(
            [
                self.build_fusion_block(ch, decoder_channels)
                for ch in reversed(fpn_channels[:-1])
            ]
        )

        self.scoring_layer = nn.Conv2d(decoder_channels, num_classes, 1)

    @abstractmethod
    def build_fusion_block(self, cin: int, cout: int) -> nn.Module: ...

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        # feats[-1] -> lowest resolution, highest semantic
        # feats[0] -> highest resolution, lowest semantic
        x = self.aspp(feats[-1])

        features_to_fuse = feats[:-1][::-1]

        # Fuse progressively higher-resolution feats into x
        for block, high_res_feat in zip(self.fusion_blocks, features_to_fuse):
            x = block(high_res_feat, x)

        prelogits = F.interpolate(
            x, size=self.input_size, mode="bilinear", align_corners=False
        )

        return self.scoring_layer(prelogits), prelogits

    @staticmethod
    def _compute_rates(height: int, stride: int) -> List[int]:
        base = max(1, height // (stride * 6))
        return [base * r for r in (1, 2, 3)]


class FaPNDecoder(BaseDecoder):
    def build_fusion_block(self, cin: int, cout: int) -> nn.Module:
        return FAM(cin, cout)


class FPNDecoder(BaseDecoder):
    def build_fusion_block(self, cin: int, cout: int) -> nn.Module:
        return ConcatFusionBlock(cin, cout)
