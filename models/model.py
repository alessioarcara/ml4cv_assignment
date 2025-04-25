from typing import Any, Dict, List

import timm
from models.decoder import DecoderWithFAM
import torch.nn as nn
from loguru import logger
from dataclasses import dataclass


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        _, logits = self.decoder(features)
        return logits


def compute_aspp_atrous_rates(image_height: int, output_stride: int) -> list[int]:
    base_rate = image_height // (output_stride * 6)
    return [base_rate * k for k in (1, 2, 3)]


@dataclass
class ModelInfo:
    fpn_features: List[int]
    atrous_rates: List[int]


def build_model(
    config: Dict[str, Any],
    num_classes: int,
    atrous_rates: List[int] | None = None,
) -> tuple[nn.Module, ModelInfo]:
    stride = config["training"]["stride"]
    imgH = config["training"]["img_height"]
    imgW = config["training"]["img_width"]

    encoder = timm.create_model(
        model_name=config["training"]["encoder_name"],
        features_only=True,
        pretrained=True,
        out_indices=(0, 1, 4),
        output_stride=stride,
    )

    fpn_features = encoder.feature_info.channels()

    if atrous_rates is None:
        atrous_rates = compute_aspp_atrous_rates(imgH, stride)

    decoder = DecoderWithFAM(
        fpn_features,
        num_classes,
        input_size=(imgH, imgW),
        d=config["training"]["d"],
        atrous_rates=atrous_rates,
    )

    model = EncoderDecoder(encoder, decoder)

    return model, ModelInfo(fpn_features, atrous_rates)
