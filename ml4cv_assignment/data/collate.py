from typing import Optional, Tuple

import torch
from torch import Tensor
from transformers import Mask2FormerImageProcessor

from ml4cv_assignment.utils.typings import Batch


def collate_fn(
    batch: Tuple[Tensor, Tensor], processor: Optional[Mask2FormerImageProcessor] = None
) -> Batch:
    images, masks = zip(*batch)

    masks_tensor = torch.stack(masks).long()

    if processor is not None:
        inputs = processor(
            images=list(images), segmentation_maps=masks_tensor, return_tensors="pt"
        )
    else:
        images_tensor = torch.stack(images)
        inputs = {"pixel_values": images_tensor}  # type: ignore

    inputs["orig_masks"] = torch.stack(masks).long()
    return inputs  # type: ignore
