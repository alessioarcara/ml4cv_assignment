from typing import Sequence
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import ASPP

class Decoder(nn.Module):
    def __init__(self, in_channels, input_size, num_classes=1, atrous_rates: Sequence[int] = (12, 24, 36), d=256):
        super().__init__()
        self.input_size = input_size
        self.aspp = ASPP(in_channels, atrous_rates)
        self.scoring_layer = nn.Conv2d(d, num_classes, 1)

    def forward(self, x):
        x = self.aspp(x)
        x = self.scoring_layer(x)
        return F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)