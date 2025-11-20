from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from ml4cv_assignment.models.model import BaseModel


class DeepLabFPN(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_lr: float,
        decoder_lr: float,
        encoder_weight_decay: float,
        decoder_weight_decay: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.encoder_weight_decay = encoder_weight_decay
        self.decoder_weight_decay = decoder_weight_decay

    def get_param_groups(self) -> List[Dict[str, Any]]:
        return [
            {
                "params": self.encoder.parameters(),
                "lr": self.encoder_lr,
                "weight_decay": self.encoder_weight_decay,
            },
            {
                "params": self.decoder.parameters(),
                "lr": self.decoder_lr,
                "weight_decay": self.decoder_weight_decay,
            },
        ]

    def forward(
        self, inputs: Dict[str, Tensor], return_preds: bool
    ) -> Dict[str, Tensor]:
        feats = self.encoder(inputs["pixel_values"])
        logits = self.decoder(feats)

        outputs_dict = {"logits": logits}

        if return_preds:
            # Since anchors are fixed orthogonal vectors minimizing Euclidean
            # distance is mathematically equivalent to maximizing the logit
            # value.
            preds = torch.argmax(logits, dim=1)

            outputs_dict["preds"] = preds

        return outputs_dict
