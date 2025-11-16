from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class DeepLabFPN(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, inputs: Dict[str, Tensor], return_preds: bool
    ) -> Dict[str, Tensor]:
        feats = self.encoder(inputs["pixel_values"])
        logits = self.decoder(feats)

        outputs_dict = {"logits": logits}

        if return_preds:
            preds = torch.argmax(logits, dim=1)
            outputs_dict["preds"] = preds

        return outputs_dict
