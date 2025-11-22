from typing import Dict, List, Optional

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

    def get_param_groups(self):
        return [
            {"params": self.model.parameters(), "lr": 1e-4, "weight_decay": 0.05},
        ]

    def _compute_segmentation_logits(
        self, outputs: Dict[str, Tensor], pixel_values: Tensor
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
        target_h, target_w = pixel_values.shape[-2:]
        mask_logits = torch.nn.functional.interpolate(
            mask_logits,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )  # [Batch, Num_Queries, H, W]
        M = mask_logits.sigmoid()

        # Multiply class probs by mask probs and sum over queries
        # b: batch, q: queries (N), c: classes (K), h: height, w: width
        L = torch.einsum("bqc, bqhw -> bchw", P, M)
        return L

    def _forward_impl(
        self, inputs: Dict[str, torch.Tensor], return_preds: bool
    ) -> Dict[str, torch.Tensor]:
        pixel_values = inputs["pixel_values"]

        mask_labels = None
        class_labels = None
        if self.should_compute_hf_loss:
            if "mask_labels" not in inputs and "class_labels" not in inputs:
                raise ValueError(
                    "To compute HuggingFace loss, 'mask_labels' and 'class_labels' must be provided in inputs."
                )
            mask_labels = inputs["mask_labels"]
            class_labels = inputs["class_labels"]

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            return_dict=True,
        )

        # Ensure outputs is a dict
        outputs_dict = dict(outputs)

        should_compute_logits = return_preds or (self.losses is not None)

        if should_compute_logits:
            L = self._compute_segmentation_logits(outputs, pixel_values)
            outputs_dict["logits"] = L

            if return_preds:
                outputs_dict["preds"] = L.argmax(dim=1)

        return outputs_dict
