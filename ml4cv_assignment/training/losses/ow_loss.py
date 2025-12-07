import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Adapted from https://github.com/PRBonn/ContMAV/blob/master/src/utils.py
class OWLoss(nn.Module):
    """
    Open World Loss (OWLoss) adapted from ContMAV (Sodano et al., CVPR 2024).

    The objective combines two distinct losses:
    1.  Feature Loss: Minimizes the distance between pixel embeddings and their class Mean Activation Vector (MAV).
        This distance is normalized by the class standard deviation to handle intra-class variance heterogeneity
        (e.g., "Sky" naturally varies more than "Road").

    2.  Contrastive Loss: An InfoNCE loss that aligns the current batch centroids with the
        historical centroids. This prevents mode collapse and ensures consistency across epochs.

    Moreover, the loss operates in two distinct phases:
    - During TRAIN (Active Phase): Accumulates running statistics (Sum, SumSq, Count)
      batch-by-batch to update class centroids (MAV) and variance at the end of the epoch.
    - During LOSS COMPUTATION (Frozen Phase): Uses the fixed centroids (MAV) and
      Standard Deviation (STD) calculated at the end of the PREVIOUS epoch to compute
      the feature loss.
    """

    mav: Tensor
    std: Tensor
    initialized: Tensor
    acc_sum: Tensor
    acc_sq_sum: Tensor
    acc_count: Tensor

    def __init__(
        self,
        num_classes: int,
        hinged: bool = False,
        delta: float = 0.1,
        ignore_index: int = 255,
        tau: float = 0.1,
        w_feat: float = 0.5,
        w_cont: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hinged = hinged
        self.delta = delta
        self.ignore_index = ignore_index
        self.tau = tau
        self.w_feat = w_feat
        self.w_cont = w_cont

        # Frozen stats from Epoch T-1 (MAV/STD)
        self.register_buffer("mav", torch.zeros(num_classes, num_classes))
        self.register_buffer("std", torch.ones(num_classes, num_classes))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.bool))

        # Epoch accumulators
        self.register_buffer(
            "acc_sum", torch.zeros(num_classes, num_classes), persistent=False
        )
        self.register_buffer(
            "acc_sq_sum", torch.zeros(num_classes, num_classes), persistent=False
        )
        self.register_buffer("acc_count", torch.zeros(num_classes), persistent=False)

    def forward(self, embeds: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            embeds: [B, C, H, W].
            targets: [B, H, W].
        """
        # Preprocessing and Flattening
        # [B, C, H, W] -> [B, H, W, C]
        embeds = embeds.permute(0, 2, 3, 1).contiguous()

        embeds_flat = embeds.view(-1, self.num_classes)  # [N_total, C]
        targets_flat = targets.view(-1)  # [N_total]

        valid_mask = targets_flat != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)

        embeds_flat = embeds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

        # We only update the running stats if the model is in training mode.
        if self.training:
            self._cumulate(embeds_flat, targets_flat)

        # If initialized is False (first epoch), we don't have valid MAVs yet.
        if not self.initialized:
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)

        # PROTOTYPE LOSS -> every pixel is attracted to its class MAV
        loss_feat = self._compute_feat_loss(embeds_flat, targets_flat)
        # CONTRASTIVE LOSS -> car cluster centroid must be far from pedestrian centroid
        loss_cont = self._compute_cont_loss(embeds_flat, targets_flat)

        return (self.w_feat * loss_feat) + (self.w_cont * loss_cont)

    @torch.no_grad()
    def _cumulate(self, embeds: Tensor, targets: Tensor) -> None:
        preds = torch.argmax(embeds, dim=1)

        # Filter: Only True Positives contribute to the class prototype
        tp_mask = preds == targets
        if not tp_mask.any():
            return

        tp_logits = embeds[tp_mask]  # [N_tp, C]
        tp_targets = targets[tp_mask]  # [N_tp]

        if tp_logits.dtype != torch.float32:
            tp_logits = tp_logits.float()

        # Accumulates sums and squared sums for König-Huygens variance calculation.
        self.acc_sum.index_add_(0, tp_targets, tp_logits)  # For E[X]
        self.acc_sq_sum.index_add_(0, tp_targets, tp_logits.pow(2))  # For E[X^2]
        self.acc_count += torch.bincount(tp_targets, minlength=self.num_classes).float()

    def _compute_feat_loss(self, embeds: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the Feature Loss weighted by Standard Deviation (Eq. 5 in the paper).
        L_feat = || f_p - mu_k || / sigma_k
        """
        # Retrieve the correct MAV and STD for each pixel based on its target class
        target_mavs = F.embedding(targets, self.mav)
        target_stds = F.embedding(targets, self.std)

        if embeds.dtype != target_mavs.dtype:
            embeds = embeds.to(target_mavs.dtype)

        # L1 Distance weighted by STD
        diff = torch.abs(embeds - target_mavs)
        weighted_dist = diff / (
            target_stds + 0.01
        )  # Aggressive clamp to avoid too small denominator

        # Sum over channels (C) to get total distance per pixel
        dist_per_pixel = weighted_dist.sum(dim=1)

        # Optional Hinge Loss
        if self.hinged:
            dist_per_pixel = F.relu(dist_per_pixel - self.delta)

        return dist_per_pixel.mean()

    def _compute_cont_loss(self, embeds: Tensor, targets: Tensor) -> Tensor:
        """
        Aligns current batch class centroids with historical centroids.
        """
        # return_indices -> returns a 1D tensor like [2,0,1,2,0,...]
        unique_labels, inverse_idx = torch.unique(targets, return_inverse=True)

        # Aggregate features to form batch centroids
        # Sums embeds into buckets based on class index (using inverse indices).
        batch_centers = torch.zeros(
            len(unique_labels),
            self.num_classes,
            device=embeds.device,
            dtype=embeds.dtype,
        )
        batch_centers.index_add_(0, inverse_idx, embeds)

        # Compute mean for each class
        counts = torch.bincount(inverse_idx).float().unsqueeze(1).to(embeds.dtype)
        batch_centers = batch_centers / counts.clamp(min=1.0)

        # Normalize f_k onto the sphere
        batch_centers = F.normalize(batch_centers, p=2, dim=1)

        # Retrieve Historical Anchors (mu)
        # These are the fixed centroids from the previous epoch.
        hist_centers = F.normalize(self.mav, p=2, dim=1)

        # Compute Similarity Matrix (Logits)
        # [K_batch, Dim] @ [Dim, K_total] -> [K_batch, K_total]
        # How similar is each current batch cluster to ALL historical clusters?
        logits = torch.matmul(batch_centers, hist_centers.T) / self.tau

        # Cross Entropy (InfoNCE)
        # Maximizes similarity to the correct historical class (diagonal),
        # Minimizes similarity to all other historical classes (off-diagonal).
        return F.cross_entropy(logits, unique_labels)

    def on_epoch_end(self) -> None:
        """
        MUST BE CALLED AT THE END OF EACH EPOCH.
        Updates the frozen MAV and STD buffers using the accumulated stats.
        """
        # Avoid division by zero
        safe_count = self.acc_count.unsqueeze(1).clamp(min=1e-8)

        # Compute New Mean (MAV) -> E[X]
        new_mav = self.acc_sum / safe_count

        # Compute New Variance -> E[X^2] - (E[X])^2
        E_x2 = self.acc_sq_sum / safe_count
        new_var = E_x2 - (new_mav**2)

        # Clamp for numerical stability (variance cannot be negative)
        new_var = torch.clamp(new_var, min=1e-8)
        new_std = torch.sqrt(new_var)

        # Update buffers
        self.mav.copy_(new_mav)
        self.std.copy_(new_std)
        self.initialized.fill_(1)

        # Reset accumulators
        self.acc_sum.zero_()
        self.acc_sq_sum.zero_()
        self.acc_count.zero_()
