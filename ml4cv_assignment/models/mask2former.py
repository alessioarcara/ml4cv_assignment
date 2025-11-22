from typing import Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    Mask2FormerForUniversalSegmentation,
)

from ml4cv_assignment.models.model import BaseModel


class Mask2Former(BaseModel):
    def __init__(self, model_id: str, num_classes: int) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(model_id)
        config.num_labels = num_classes

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id, config=config, ignore_mismatched_sizes=True
        )

    def get_param_groups(self):
        return [
            {"params": self.model.parameters(), "lr": 1e-4, "weight_decay": 0.05},
        ]

    def forward(
        self, inputs: Dict[str, torch.Tensor], return_preds: bool
    ) -> Dict[str, torch.Tensor]:
        pixel_values = inputs["pixel_values"]
        mask_labels = inputs.get("mask_labels", None)
        class_labels = inputs.get("class_labels", None)

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            return_dict=True,
        )

        if return_preds:
            # Shape: [Batch, Num_Queries, Num_Classes + 1] (including 'no object' class)
            class_logits = outputs.class_queries_logits
            # Shape: [Batch, Num_Queries, H/4, W/4]
            mask_logits = outputs.masks_queries_logits

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

            outputs["preds"] = L.argmax(dim=1)

        return outputs
