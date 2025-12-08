import torch
import torch.nn as nn
from torch import Tensor


class OoDBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss for OoD Head.

    Treats the problem as binary classification:
    - Known classes are labeled as 0
    - Unknown classes are labeled as 1
    """

    def __init__(self, unknown_label: int = -1, ignore_index: int = 255) -> None:
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.unknown_label = unknown_label
        self.ignore_index = ignore_index

    def forward(
        self,
        ood_logits: Tensor,  # [B, 1, H, W] - logits from OoD head
        labels: Tensor,  # [B, H, W]
    ) -> Tensor:
        if ood_logits.dim() == 4:
            ood_logits = ood_logits.squeeze(1)  # [B, H, W]

        is_ignore = labels == self.ignore_index
        is_unknown = labels == self.unknown_label

        targets = is_unknown.float()

        loss = self.bce_loss(ood_logits.squeeze(1), targets)

        loss = loss * (~is_ignore).float()

        valid_pixel_count = (~is_ignore).sum()

        if valid_pixel_count == 0:
            return torch.tensor(0.0, device=ood_logits.device, requires_grad=True)

        return loss.sum() / valid_pixel_count
