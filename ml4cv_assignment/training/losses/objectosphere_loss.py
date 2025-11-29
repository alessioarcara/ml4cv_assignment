import torch
import torch.nn as nn
from torch import Tensor


class ObjectosphereLoss(nn.Module):
    def __init__(
        self, xi: float = 1.0, unknown_label: int = -1, ignore_index: int = 255
    ):
        super().__init__()
        self.xi = xi
        self.unknown_label = unknown_label
        self.ignore_index = ignore_index

    def forward(
        self,
        feats: Tensor,  # [B, C, H, W]
        labels: Tensor,  # [B, H, W]
    ):
        # Compute norm for each pixel
        norm = torch.linalg.norm(feats, ord=2, dim=1)  # [B, H, W]

        is_ignore = labels == self.ignore_index
        is_unknown = labels == self.unknown_label
        is_known = ~(is_ignore | is_unknown)

        losses = torch.zeros_like(norm, dtype=norm.dtype)

        # Known classes: Push feature OUTSIDE the sphere of radius 'xi' (Norm > xi)
        if is_known.any():
            losses[is_known] = torch.relu(self.xi - norm[is_known]).pow(2)

        # Unknown classes: Pull feature towards the ORIGIN (0) (Norm -> 0)
        if is_unknown.any():
            losses[is_unknown] = norm[is_unknown].pow(2)

        valid_pixel_count = (~is_ignore).sum()

        if valid_pixel_count == 0:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)

        return losses.sum() / valid_pixel_count
