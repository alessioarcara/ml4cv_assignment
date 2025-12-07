from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ml4cv_assignment.training.losses.base_pixel_metric_learning_loss import (
    BasePixelMetricLearningLoss,
)


# Adapted from https://github.com/Brilhador/tgrs2023/blob/main/utils/losses.py
class PrototypicalGlobalLocalTripletLoss(BasePixelMetricLearningLoss):
    """ """

    anchors: Tensor

    def __init__(
        self,
        num_classes: int,
        margin_global: float,
        margin_local: float,
        magnitude: int = 3,
        ignore_index: int = 255,
    ):
        super().__init__(num_classes, magnitude, ignore_index)

        self.pdist = nn.PairwiseDistance(p=2)

        self.triplet_loss_global = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist, swap=False, margin=margin_global
        )
        self.triplet_loss_local = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist, swap=False, margin=margin_local
        )

    def _sample_triplets(
        self,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Samples triplets from the given anchors, positives and negatives tensors.
        Returns empty tensors if no triplets can be formed.
        """
        num_anchors, num_positives, num_negatives = (
            anchors.size(0),
            positives.size(0),
            negatives.size(0),
        )
        num_triplets = min(num_anchors, num_positives, num_negatives)

        if num_triplets == 0:
            return anchors[:0], positives[:0], negatives[:0]

        def get_indices(size: int) -> Tensor:
            return torch.randperm(size, device=anchors.device)[:num_triplets]

        anchors_sampled = anchors[get_indices(num_anchors)]
        positives_sampled = positives[get_indices(num_positives)]
        negatives_sampled = negatives[get_indices(num_negatives)]

        return anchors_sampled, positives_sampled, negatives_sampled

    def _compute_global_triplet_loss(
        self,
        embeds: Tensor,
        targets: Tensor,
        preds: Tensor,
    ) -> Tensor:
        """
        Global: pull true positives towards class anchors, push false positives away from class anchors.
        """
        embeds_global, targets_global, preds_global = self.preprocess_inputs(
            embeds, targets, preds
        )  # [N_valid, C], [N_valid], [N_valid]

        triplet_loss_global = torch.tensor(0.0, device=embeds.device)
        valid_global_classes = 0

        # We iterate over anchors, not just present classes
        # because the model might have predicted a class that isn't present in the ground truth
        for c, anchor in enumerate(self.anchors):
            # positives (TP) (Easy positives)
            # We want to pull correct classified pixels towards the anchor of their class
            positive_feats = embeds_global[(preds_global == c) & (targets_global == c)]
            # negative (FP) (Hard negatives)
            # We want to push wrong classified pixels away from the anchor of their class
            negative_feats = embeds_global[(preds_global == c) & (targets_global != c)]

            if positive_feats.size(0) == 0 or negative_feats.size(0) == 0:
                continue

            num_triplets = min(positive_feats.size(0), negative_feats.size(0))

            p_s = positive_feats[
                torch.randperm(positive_feats.size(0), device=positive_feats.device)[
                    :num_triplets
                ]
            ]
            n_s = negative_feats[
                torch.randperm(negative_feats.size(0), device=negative_feats.device)[
                    :num_triplets
                ]
            ]
            a_s = anchor.unsqueeze(0).expand(num_triplets, -1)

            triplet_loss_global += self.triplet_loss_global(a_s, p_s, n_s)
            valid_global_classes += 1

        if valid_global_classes > 0:
            triplet_loss_global /= valid_global_classes

        return triplet_loss_global

    def _compute_local_triplet_loss(
        self,
        embeds: Tensor,
        targets: Tensor,
        preds: Tensor,
    ) -> Tensor:
        """
        Local: pull true positives towards each other, push false negatives away from true positives.
        """
        embeds = embeds.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        B = embeds.shape[0]

        embeds_local = embeds.view(B, -1, self.num_classes)  # [B, H*W, C]
        targets_local = targets.view(B, -1)  # [B, H*W]
        preds_local = preds.view(B, -1)  # [B, H*W]

        local_loss = torch.tensor(0.0, device=embeds.device)
        valid_images_count = 0

        # Loop images in batch (to compute local attraction and repulsion)
        for b in range(B):
            # Filter valid pixels: only consider pixels that are not 'ignore_index'
            valid_mask = targets_local[b] != self.ignore_index

            embeds_img = embeds_local[b][valid_mask]  # [N_valid, C]
            targets_img = targets_local[b][valid_mask]  # [N_valid]
            preds_img = preds_local[b][valid_mask]  # [N_valid]

            loss_image = torch.tensor(0.0, device=embeds.device)
            valid_classes_count = 0

            present_classes = torch.unique(targets_img)

            for class_id in present_classes:
                is_current_class = targets_img == class_id
                is_predicted_class = preds_img == class_id

                # Positives (TP) (Easy local triplets positives)
                anchor_feats = embeds_img[is_current_class & is_predicted_class]
                # Positives (FN) (False negatives are hard local triplets positives)
                positive_feats = embeds_img[is_current_class & (~is_predicted_class)]
                # Negatives (TN + FP)
                negative_feats = embeds_img[~is_current_class]

                a_s, p_s, n_s = self._sample_triplets(
                    anchor_feats, positive_feats, negative_feats
                )

                if a_s.size(0) > 0:
                    loss_image += self.triplet_loss_local(a_s, p_s, n_s)
                    valid_classes_count += 1

            if valid_classes_count > 0:
                local_loss += loss_image / valid_classes_count
                valid_images_count += 1

        if valid_images_count > 0:
            local_loss /= valid_images_count

        return local_loss

    def forward(self, embeds: Tensor, targets: Tensor) -> Tensor:
        """
        embeds: [B, C, H, W] predicted embeddings
        targets: [B, H, W] ground-truth labels
        """
        preds = torch.argmax(embeds, dim=1)  # [B, H, W]

        loss_local = self._compute_local_triplet_loss(embeds, targets, preds)
        loss_global = self._compute_global_triplet_loss(embeds, targets, preds)

        return loss_global + loss_local
