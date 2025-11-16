from typing import List, Optional, Sequence, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ml4cv_assignment.models.layers.aspp import ASPP
from ml4cv_assignment.models.layers.fapn import FAM


class FaPNDecoder(nn.Module):
    def __init__(
        self,
        fpn_channels: List[int],
        num_classes: int,
        decoder_channels: int,
        input_size: Tuple[int, int],
        output_stride: int,
        atrous_rates: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.decoder_channels = decoder_channels

        if atrous_rates is None:
            atrous_rates = self.compute_aspp_atrous_rates(
                image_height=input_size[0], output_stride=output_stride
            )

        self.aspp = ASPP(fpn_channels[-1], atrous_rates, decoder_channels)

        self.fam_blocks = nn.ModuleList(
            [FAM(ch, decoder_channels) for ch in reversed(fpn_channels[:-1])]
        )

        self.scoring_layer = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, feats: Tuple[Tensor]):
        decoder_feature = self.aspp(feats[-1])

        # Fuse progressively higher-resolution features into the decoder feature
        for fam_block, high_res_feature in zip(self.fam_blocks, feats[:-1][::-1]):  # type: ignore
            decoder_feature = fam_block(high_res_feature, decoder_feature)

        prelogits = F.interpolate(
            decoder_feature, size=self.input_size, mode="bilinear", align_corners=False
        )
        logits = self.scoring_layer(prelogits)
        return logits

    @staticmethod
    def compute_aspp_atrous_rates(image_height: int, output_stride: int) -> List[int]:
        base_dilation = max(1, image_height // (output_stride * 6))
        dilation_rates = [base_dilation * factor for factor in (1, 2, 3)]
        return dilation_rates
