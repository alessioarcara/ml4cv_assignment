from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from transformers import (
    AutoConfig,
    Mask2FormerForUniversalSegmentation,
)

from ml4cv_assignment.models.base_model import BaseModel
from ml4cv_assignment.models.layers.modeling_mask2former import (
    Mask2FormerPixelDecoder,
)


class Mask2Former(BaseModel):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        lr: float,
        weight_decay: float,
        losses: Optional[List[torch.nn.Module]] = None,
        freeze_all_except_heads: bool = False,
        should_compute_hf_loss: bool = True,
        use_rba_decoder: bool = False,
        decoder_num_layers: int = 2,
    ) -> None:
        super().__init__(losses=losses)
        self.lr = float(lr)
        self.weight_decay = weight_decay
        self.should_compute_hf_loss = should_compute_hf_loss

        config = AutoConfig.from_pretrained(model_id)
        config.num_labels = num_classes
        config.pre_norm = True
        config.decoder_layers = decoder_num_layers

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id, config=config, ignore_mismatched_sizes=True
        )

        if use_rba_decoder:
            # Remove HF PixelDecoder and replace with custom one from RbA paper
            original_decoder = self.model.model.pixel_level_module.decoder
            feature_channels = original_decoder.feature_channels

            custom_decoder = Mask2FormerPixelDecoder(config, feature_channels)

            custom_decoder.load_state_dict(original_decoder.state_dict(), strict=False)

            self.model.model.pixel_level_module.decoder = custom_decoder
            logger.info("Using RbA Custom PixelDecoder")

            self.model.model.transformer_module.num_feature_levels = 1
            self.model.model.transformer_module.decoder.num_feature_levels = 1
        else:
            logger.info("Using Standard HuggingFace PixelDecoder")

        # We finetune only the mask-prediction MLP and the post-decoder classification layer
        # to preserve the model closed-set performance.
        if freeze_all_except_heads:
            self.freeze_module(self.model)
            self.unfreeze_module(
                self.model.model.transformer_module.decoder.mask_predictor
            )
            self.unfreeze_module(self.model.class_predictor)

    # override
    def get_param_groups(self):
        backbone = self.model.model.pixel_level_module.encoder
        backbone_params = list(backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}

        trainable_backbone_params = [
            p for p in backbone.parameters() if p.requires_grad
        ]

        trainable_other_params = [
            p
            for p in self.model.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]

        param_groups = []
        if trainable_backbone_params:
            param_groups.append(
                {
                    "params": trainable_backbone_params,
                    "lr": self.lr * 0.1,
                    "weight_decay": self.weight_decay,
                }
            )

        if trainable_other_params:
            param_groups.append(
                {
                    "params": trainable_other_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            )
        return param_groups

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
                ood_score = -L.tanh().sum(dim=1)
                outputs_dict["ood_score"] = ood_score

        return outputs_dict
