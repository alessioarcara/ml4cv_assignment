from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Model(nn.Module):
    def __init__(self, model: nn.Module, losses: List[nn.Module]) -> None:
        super().__init__()
        self.model = model
        self.losses = losses

    def _compute_loss(
        self, logits: Tensor, gt_masks: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        losses = []
        loss_dict = {}

        for i, loss_fn in enumerate(self.losses):
            loss_i: Tensor = loss_fn(logits, gt_masks)
            losses.append(loss_i)
            loss_dict[f"batch_loss_{i}"] = loss_i.item()

        total_loss = torch.mean(torch.stack(losses))

        return total_loss, loss_dict

    def forward(self, inputs: dict, return_preds: bool = True) -> Dict[str, Any]:
        outputs = self.model(inputs, return_preds=return_preds)

        # Compute loss only if:
        # - losses are defined
        # - loss is not already computed in outputs
        # - model outputs logits
        if self.losses and "logits" in outputs and "loss" not in outputs:
            targets = inputs.get("orig_masks")
            if targets is not None:
                total_loss, loss_dict = self._compute_loss(
                    outputs["logits"], inputs["orig_masks"]
                )
                outputs["loss"] = total_loss
                outputs.update(loss_dict)

        return outputs
