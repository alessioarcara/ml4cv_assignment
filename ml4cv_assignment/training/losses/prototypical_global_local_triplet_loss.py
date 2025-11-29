from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


# Adapted from https://github.com/Brilhador/tgrs2023/blob/main/utils/losses.py
class PrototypicalGlobalLocalTripletLoss(nn.Module):
    anchors: Tensor

    def __init__(
        self,
        num_classes: int,
        margin_global: float,
        margin_local: float,
        magnitude: int = 3,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.magnitude = magnitude
        self.ignore_index = ignore_index

        self.build_anchors()

        self.pdist = nn.PairwiseDistance(p=2)

        self.triplet_loss_global = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist, swap=False, margin=margin_global
        )
        self.triplet_loss_local = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist, swap=False, margin=margin_local
        )

    def build_anchors(self) -> None:
        """
        Creates fixed anchors scaled by magnitude
        """
        anchors = torch.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            anchors[i][i] = self.magnitude
        self.register_buffer("anchors", anchors)

    def _sample_triplets(
        self,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Samples triplets from the given anchors, positives, and negatives tensors.
        Returns empty tensors if no triplets can be formed.
        """
        num_triplets = min(anchors.size(0), positives.size(0), negatives.size(0))

        if num_triplets == 0:
            return anchors[:0], positives[:0], negatives[:0]

        anchors_sampled = anchors[
            torch.randperm(anchors.size(0), device=anchors.device)
        ][:num_triplets]
        positives_sampled = positives[
            torch.randperm(positives.size(0), device=positives.device)
        ][:num_triplets]
        negatives_sampled = negatives[
            torch.randperm(negatives.size(0), device=negatives.device)
        ][:num_triplets]

        return anchors_sampled, positives_sampled, negatives_sampled

    def forward(self, embeds: Tensor, targets: Tensor) -> Tensor:
        """
        embeds: [B, C, H, W] predicted embeddings
        targets: [B, H, W] ground-truth labels
        """
        embeds = embeds.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        preds = torch.argmax(embeds, dim=3)  # [B, H, W]

        B, _, _, C = embeds.shape

        # Flatten spatial dimensions
        embeds_flat = embeds.view(B, -1, C)  # [B, H*W, C]
        targets_flat = targets.view(B, -1)  # [B, H*W]
        preds_flat = preds.view(B, -1)  # [B, H*W]

        ###################################
        # LOCAL ATTRACTION AND REPULSION #
        ###################################

        triplet_loss_local = torch.tensor(0.0, device=embeds.device)
        valid_images_count = 0

        # Loop images in batch (to compute local attraction and repulsion)
        for b in range(B):
            # Filter valid pixels: only consider pixels that are not 'ignore_index'
            valid_mask = targets_flat[b] != self.ignore_index

            embeds_img = embeds_flat[b][valid_mask]  # [N_valid, C]
            targets_img = targets_flat[b][valid_mask]  # [N_valid]
            preds_img = preds_flat[b][valid_mask]  # [N_valid]

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
                triplet_loss_local += loss_image / valid_classes_count
                valid_images_count += 1

        if valid_images_count > 0:
            triplet_loss_local /= valid_images_count

        ###################################
        # GLOBAL ATTRACTION AND REPULSION #
        ###################################

        embeds_global = embeds.view(-1, C)  # [B*H*W, C]
        targets_global = targets.view(-1)  # [B*H*W]
        preds_global = preds.view(-1)  # [B*H*W]

        valid_mask_global = targets_global != self.ignore_index

        embeds_global = embeds_global[valid_mask_global]
        targets_global = targets_global[valid_mask_global]
        preds_global = preds_global[valid_mask_global]

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

        return triplet_loss_global + triplet_loss_local
