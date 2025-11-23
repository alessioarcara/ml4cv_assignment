from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoConfig,
    Mask2FormerForUniversalSegmentation,
)

from ml4cv_assignment.models.base_model import BaseModel


class Mask2Former(BaseModel):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        losses: Optional[List[torch.nn.Module]] = None,
        freeze_all_except_heads: bool = False,
        should_compute_hf_loss: bool = True,
    ) -> None:
        super().__init__(losses=losses)
        self.should_compute_hf_loss = should_compute_hf_loss

        config = AutoConfig.from_pretrained(model_id)
        config.num_labels = num_classes

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id, config=config, ignore_mismatched_sizes=True
        )

        # We finetune only the mask-prediction MLP and the post-decoder classification layer
        # to preserve the model closed-set performance.
        if freeze_all_except_heads:
            for p in self.model.parameters():
                p.requires_grad = False

            for p in (
                self.model.model.transformer_module.decoder.mask_predictor.parameters()
            ):
                p.requires_grad = True

            for p in self.model.class_queries_logits.parameters():
                p.requires_grad = True

    # override
    def get_param_groups(self):
        return [
            {"params": self.model.parameters(), "lr": 1e-4, "weight_decay": 0.05},
        ]

    def _compute_segmentation_logits(
        self, outputs: Dict[str, Tensor], target_hw: Tuple[int, int]
    ) -> torch.Tensor:
        # Shape: [Batch, Num_Queries, Num_Classes + 1] (including 'no object' class)
        class_logits = outputs["class_queries_logits"]
        # Shape: [Batch, Num_Queries, H/4, W/4]
        mask_logits = outputs["masks_queries_logits"]

        # Remove the last class (the 'no object' class)
        P = F.softmax(class_logits, dim=-1)[
            ..., :-1
        ]  # [Batch, Num_Queries, Num_Classes]

        # Upsample N masks to full image resolution (expensive in terms of memory)
        mask_logits = torch.nn.functional.interpolate(
            mask_logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )  # [Batch, Num_Queries, H, W]
        M = mask_logits.sigmoid()

        # Multiply class probs by mask probs and sum over queries
        # b: batch, q: queries (N), c: classes (K), h: height, w: width
        L = torch.einsum("bqc, bqhw -> bchw", P, M)
        return L

    # override
    def _forward_impl(
        self, inputs: Dict[str, torch.Tensor], return_preds: bool
    ) -> Dict[str, torch.Tensor]:
        pixel_values = inputs["pixel_values"]
        mask_labels = inputs.get("mask_labels")
        class_labels = inputs.get("class_labels")

        if self.should_compute_hf_loss and (
            mask_labels is None or class_labels is None
        ):
            raise ValueError(
                "For HF loss, 'mask_labels' and 'class_labels' are required."
            )

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels if self.should_compute_hf_loss else None,
            class_labels=class_labels if self.should_compute_hf_loss else None,
            return_dict=True,
        )

        # Ensure outputs is a dict
        outputs_dict = dict(outputs)

        should_compute_logits = return_preds or (len(self.losses) > 0)

        if should_compute_logits:
            target_hw = (pixel_values.shape[-2], pixel_values.shape[-1])
            # Shape: [Batch, Num_Classes, H, W]
            L = self._compute_segmentation_logits(outputs_dict, target_hw)
            outputs_dict["logits"] = L

            if return_preds:
                # ID prediction: argmax over known classes
                outputs_dict["preds"] = L.argmax(dim=1)

                # OOD prediction (RBA): 1- total activation
                # High score means the pixel is "rejected" (low activation) by all known classes
                ood_score = 1 - L.tanh().sum(dim=1)
                outputs_dict["ood_score"] = ood_score

        return outputs_dict
