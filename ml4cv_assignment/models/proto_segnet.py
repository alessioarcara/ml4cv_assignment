from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ml4cv_assignment.models.base_model import BaseModel


class RunningCenters(nn.Module):
    running_centers: Tensor

    def __init__(self, num_classes: int, momentum: float):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum

        self.register_buffer("running_centers", torch.zeros(num_classes, num_classes))

    @property
    def centers(self) -> Tensor:
        return self.running_centers

    @torch.no_grad()
    def update(
        self,
        embeds: Tensor,  # [N, C]
        labels: Tensor,  # [N]
    ) -> None:
        # Filter out invalid labels
        valid = (labels >= 0) & (labels < self.num_classes)
        embeds, labels = embeds[valid], labels[valid]

        embeds = embeds.to(dtype=self.running_centers.dtype)

        unique_classes = labels.unique()

        for cls in unique_classes:
            cls_mask = labels == cls

            # Compute mean batch center for class `cls`
            batch_center = embeds[cls_mask].mean(dim=0)  # [C]

            running_center = self.running_centers[cls]  # [C]

            # Check if running center is uninitialized
            if running_center.abs().sum() == 0:
                running_center.copy_(batch_center)
            else:
                # EMA Update:
                # Formula: new_val = momentum * old_val + (1 - momentum) * batch_val
                # PyTorch `lerp_` performs: input + weight * (end - input)
                # So we use weight = (1 - momentum)
                running_center.lerp_(batch_center, weight=1 - self.momentum)

    def reset(self) -> None:
        self.running_centers.zero_()


class ProtoSegNet(BaseModel):
    anchors: Tensor
    fixed_anchors: Tensor

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encoder_lr: float,
        decoder_lr: float,
        encoder_weight_decay: float,
        decoder_weight_decay: float,
        num_classes: int,
        T: float,
        alpha: float,
        xi: float,
        anchors_magnitude: float,
        use_running_centers: bool,
        centers_momentum: float,
        losses: Optional[List[nn.Module]] = None,
    ) -> None:
        super().__init__(losses=losses)

        self.encoder = encoder
        self.decoder = decoder

        # Optimizer hyperparameters
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.encoder_weight_decay = encoder_weight_decay
        self.decoder_weight_decay = decoder_weight_decay

        # OoD hyperparameters
        self.T = T
        self.alpha = alpha
        self.xi = xi
        self.use_running_centers = use_running_centers

        # fixed anchors shouldn't be saved in the state dict
        # they are only used to reset the anchors at the start of each epoch
        initial = torch.eye(num_classes) * anchors_magnitude
        self.register_buffer("anchors", initial)
        self.register_buffer("fixed_anchors", initial.clone(), persistent=False)

        if self.use_running_centers:
            self.running_centers = RunningCenters(num_classes, centers_momentum)

    # override
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

    def _compute_ood_scores(self, embeds: Tensor, dists: Tensor) -> Tensor:
        """
        Args:
            embeds: [B*H*W, C] - pixel embeddings
            dists: [B*H*W, K] - distance of each pixel to each prototype
        Returns:
            ood_score: [B*H*W] - OoD score for each pixel
        """
        # 1. Feature-Norm Score
        # If the feature norm is low (close to 0), it indicates the network
        # has mapped the input to the "unknown" region (origin).
        # Logic: low norm -> high ood score
        norms_sq = embeds.norm(p=2, dim=1).pow(2)  # [B*H*W]
        feat_score = (1.0 - norms_sq / self.xi).clamp(min=0.0)  # [B*H*W]

        # 2. Prototype Distance Score
        # Open Space Risk: An input is OOD if it is EITHER far from all centers (Distance)
        # OR equidistant between centers (Uncertainty).
        softmin = torch.softmax(-dists / self.T, dim=1)  # [B*H*W, K]
        gamma = dists * (1 - softmin)
        cac_score, _ = gamma.min(dim=1)  # [B*H*W]

        # I apply tanh to squash unbounded distance score into [0, 1]
        normalized_cac_score = cac_score.tanh()

        return self.alpha * feat_score + (1 - self.alpha) * normalized_cac_score

    # override
    def _forward_impl(
        self, inputs: Dict[str, Tensor], return_preds: bool
    ) -> Dict[str, Tensor]:
        feats = self.encoder(inputs["pixel_values"])
        logits = self.decoder(feats)
        outputs_dict = {"logits": logits}

        needs_flat_processing = return_preds or (
            self.training and self.use_running_centers
        )

        if not needs_flat_processing:
            return outputs_dict

        B, C, H, W = logits.shape
        embeds = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Update centers
        if self.training and self.use_running_centers:
            self.running_centers.update(embeds, inputs["orig_masks"].view(-1))

        # Inference / Metrics
        if return_preds:
            dists = torch.cdist(embeds, self.anchors, p=2)  # [B*H*W, C]

            # If using fixed orthogonal anchors, Max Logit is equivalent to Min Distance.
            # If using Running Centers (which might not be orthogonal),
            # we must use Min Distance to find the nearest center.
            if self.use_running_centers:
                preds = torch.argmin(dists, dim=1).view(B, H, W)
            else:
                preds = torch.argmax(logits, dim=1)

            outputs_dict["preds"] = preds

            flat_ood_scores = self._compute_ood_scores(embeds, dists)
            outputs_dict["ood_score"] = flat_ood_scores.view(B, H, W)

        return outputs_dict

    def on_train_epoch_start(self) -> None:
        """
        Recover fixed anchors to guarantee stability on training.
        """
        if self.use_running_centers:
            self.anchors.copy_(self.fixed_anchors)

    def on_eval_start(self) -> None:
        if self.use_running_centers:
            self.anchors.copy_(self.running_centers.centers)
