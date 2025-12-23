import torch.nn as nn
from torch import Tensor


class UnknownAwareLoss(nn.Module):
    def __init__(
        self, loss: nn.Module, unknown_label: int = -1, ignore_index: int = 255
    ) -> None:
        """
        Wraps a loss function to handle 'unknown' labels by mapping them to ignore_index.
        """
        super().__init__()
        self.loss = loss
        self.unknown_label = unknown_label
        self.ignore_index = ignore_index

        base_name = getattr(loss, "__name__", type(loss).__name__)
        self.name = f"UnknownAware_{base_name}"

    def forward(self, logits: Tensor, gt_masks: Tensor) -> Tensor:
        targets = gt_masks.clone()
        targets[gt_masks == self.unknown_label] = self.ignore_index
        return self.loss(logits, targets)
