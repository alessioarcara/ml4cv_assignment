import torch
import torch.nn as nn
from torch import Tensor

from ml4cv_assignment.utils.typings import OptTensor


class BasePixelMetricLearningLoss(nn.Module):
    def __init__(self, num_classes: int, magnitude: float, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.magnitude = magnitude
        self.ignore_index = ignore_index

        self.register_buffer("anchors", torch.eye(self.num_classes) * self.magnitude)

    def forward(self, embeds: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    def preprocess_inputs(
        self, embeds: Tensor, targets: Tensor, preds: OptTensor = None
    ) -> tuple[Tensor, Tensor, OptTensor]:
        """
        Flattens inputs (B, C, H, W -> N, C) and filters 'ignore_index'.
        Returns: (embeds_flat, targets_flat)
        """
        # [B, C, H, W] -> [B, H, W, C]
        embeds = embeds.permute(0, 2, 3, 1).contiguous()

        # Flatten spatial dims: [N_total, C] and [N_total]
        embeds_flat = embeds.view(-1, self.num_classes)
        targets_flat = targets.view(-1)
        preds_flat = preds.view(-1) if preds is not None else None

        # Filter invalid pixels
        valid_mask = targets_flat != self.ignore_index
        return (
            embeds_flat[valid_mask],
            targets_flat[valid_mask],
            preds_flat[valid_mask] if preds_flat is not None else None,
        )
