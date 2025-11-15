import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)


class Mask2Former(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model_id = "facebook/mask2former-swin-tiny-cityscapes-semantic"

        config = AutoConfig.from_pretrained(model_id)
        config.num_labels = 13

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id, config=config, ignore_mismatched_sizes=True
        )

        self.processor = Mask2FormerImageProcessor(
            do_resize=False,
            do_normalize=False,
            num_labels=13,
            ignore_index=255,
        )

    def forward(
        self, inputs: dict[str, torch.Tensor], return_preds: bool = True
    ) -> dict[str, torch.Tensor]:
        pixel_values = inputs["pixel_values"]
        mask_labels = inputs.get("mask_labels", None)
        class_labels = inputs.get("class_labels", None)

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        outputs_dict = dict(outputs)

        if return_preds:
            preds = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[pixel_values.shape[-2:]] * pixel_values.shape[0]
            )
            outputs_dict["preds"] = torch.stack(preds)

        return outputs_dict
