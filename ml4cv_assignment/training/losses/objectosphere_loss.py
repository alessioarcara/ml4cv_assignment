import torch
import torch.nn as nn
from torch import Tensor


class ObjectosphereLoss(nn.Module):
    def __init__(self, xi: float = 1.0, unknown_label: int = -1):
        super(ObjectosphereLoss, self).__init__()
        self.xi = xi
        self.unknown_label = unknown_label

    def forward(
        self,
        feats: Tensor,  # [B, C, H, W]
        labels: Tensor,  # [B, H, W]  (long / int)
    ):
        norm = torch.linalg.norm(feats, ord=2, dim=1)

        unknown = labels == self.unknown_label
        known = ~unknown

        losses = torch.zeros_like(norm, dtype=norm.dtype)
        losses[known] = torch.relu(self.xi - norm[known]).pow(2)
        losses[unknown] = norm[unknown].pow(2)

        return losses.mean()
