import torch
import torch.nn.functional as F
from torch import Tensor

from ml4cv_assignment.training.losses.base_pixel_metric_learning_loss import (
    BasePixelMetricLearningLoss,
)


class DMLLoss(BasePixelMetricLearningLoss):
    """
    Loss derived from the paper 'Deep Metric Learning for Open World Semantic Segmentation'
    Combines Discriminative Cross Entropy and Variance Loss.

    THEORY:

    1. STANDARD CE vs DISTANCE CE
       * Standard CE (Score = W*x): Based on Dot Product. It depends only on the ANGLE (direction).
         It divides the space into infinite "cones". A very distant "Unknown" object,
         if aligned with a class, receives a high score. Fails in Open World settings.
       * Distance CE (Logit = -||x - c||^2): Based on Euclidean Distance. It depends on the POSITION.
         It creates closed regions ("spheres"). If an object is far, the score drops drastically.

    2. WHY VARIANCE LOSS IS NEEDED (Intra-class Compactness)
       * DCE uses Softmax, which looks only at relative distances (Inter-class).
         If dist_A = 20 and dist_B = 100, the CE is satisfied (loss ~ 0) because A wins
         easily against B. However, a distance of 20 is geometrically huge!
         The cluster remains "sparse" and wide.
       * Variance Loss forces the absolute distance to zero, ignoring other classes.
         It constrains all points of a class to form a dense "ball"
         around the fixed centroid.
    """

    anchors: Tensor

    def __init__(
        self,
        num_classes: int,
        magnitude: float = 3.0,
        alpha: float = 0.01,
        ignore_index: int = 255,
    ):
        super().__init__(num_classes, magnitude, ignore_index)
        self.alpha = alpha

    def forward(
        self,
        embeds: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        embeds: [B, C, H, W] predicted embeddings
        targets: [B, H, W] ground-truth labels
        """
        embeds_flat, targets_flat, _ = self.preprocess_inputs(embeds, targets)

        dists = torch.cdist(embeds_flat, self.anchors, p=2)  # [N, C]
        dists_sq = dists.pow(2)

        # ----------------------------------------
        # Discriminative Cross Entropy
        # ----------------------------------------
        # Logits are negative distances (closer = higher prob)
        logits = -dists_sq  # [N, C]
        loss_dce = F.cross_entropy(logits, targets_flat)

        # ----------------------------------------
        # Variance Loss
        # ----------------------------------------
        # Get the distance to the correct class for each pixel
        # gather needs indices of shape [N, 1]
        distance_to_target = dists_sq.gather(1, targets_flat.unsqueeze(1)).squeeze(
            1
        )  # [N]
        loss_var = distance_to_target.mean()

        total_loss = loss_dce + (self.alpha * loss_var)

        return total_loss
