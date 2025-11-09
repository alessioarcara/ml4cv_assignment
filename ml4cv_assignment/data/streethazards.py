from pathlib import Path
from typing import Tuple, Union

import cv2 as cv
import numpy as np
import torch
from pytorch_ood.augment import InsertCOCO
from torch import Tensor
from torch.utils.data import Dataset

from ml4cv_assignment.data.torch_serialized_list import TorchSerializedList
from ml4cv_assignment.utils.typings import PathOrStr


class StreetHazards(Dataset):
    def __init__(
        self,
        root_dir: PathOrStr,
        subset: str = "training",
        transforms=None,
        add_anomalies: bool = False,
    ) -> None:
        root_dir = Path(root_dir)
        images_dir = root_dir / "images" / subset
        masks_dir = root_dir / "annotations" / subset

        self.imgs = TorchSerializedList(
            sorted(str(p) for p in images_dir.rglob("*.png"))
        )
        self.masks = TorchSerializedList(
            sorted(str(p) for p in masks_dir.rglob("*.png"))
        )
        self.transforms = transforms

        self.coco_transform = (
            InsertCOCO(
                coco_dir="data/datasets/coco/",
                exclude_classes="Streethazards",
                p=1,
                ood_mask_value=14,
            )
            if add_anomalies
            else None
        )

        if len(self.imgs) != len(self.masks):
            raise ValueError(
                f"Number of images ({len(self.imgs)}) and masks ({len(self.masks)}) do not match."
            )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(
        self, idx: int, apply_transforms: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        img = cv.imread(self.imgs[idx], cv.IMREAD_COLOR_RGB)
        mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)

        if self.coco_transform is not None:
            img, mask = self.coco_transform(img, mask)
            assert isinstance(mask, torch.Tensor), "InsertCOCO returns mask as Tensor"
            mask = mask.cpu().numpy()

        if apply_transforms and self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        assert img is not None and mask is not None, (
            "Image or mask is None after transformations"
        )
        return img, (mask - 1)
