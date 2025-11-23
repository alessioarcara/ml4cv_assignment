import torch
import torch.nn as nn
from torch import Tensor


class ObjectosphereLoss(nn.Module):
    def __init__(self, xi: float = 1.0, unknown_label: int = -1):
        super().__init__()
        self.xi = xi
        self.unknown_label = unknown_label

    def forward(
        self,
        feats: Tensor,  # [B, C, H, W]
        labels: Tensor,  # [B, H, W]
    ):
        norm = torch.linalg.norm(feats, ord=2, dim=1)

        unknown = labels == self.unknown_label
        known = ~unknown

        losses = torch.zeros_like(norm, dtype=norm.dtype)

        # Known classes: Push feature OUTSIDE the sphere of radius 'xi'
        losses[known] = torch.relu(self.xi - norm[known]).pow(2)

        # Unknown classes: Pull feature towards the ORIGIN (0)
        losses[unknown] = norm[unknown].pow(2)

        return losses.mean()
