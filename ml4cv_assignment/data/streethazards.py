from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2 as cv
import numpy as np
import torch
from pytorch_ood.augment import InsertCOCO
from torch import Tensor
from torch.utils.data import Dataset
from transformers import Mask2FormerImageProcessor

from ml4cv_assignment.config.dataset_config import StreetHazardsDatasetConfig
from ml4cv_assignment.data.data_utils import TorchSerializedList


# class StreetHazards(Dataset):
#    CLASSES = [
#        "unlabeled",
#        "building",
#        "fence",
#        "other",
#        "pedestrian",
#        "pole",
#        "road line",
#        "road",
#        "sidewalk",
#        "vegetation",
#        "car",
#        "wall",
#        "traffic sign",
#        "anomaly",
#    ]
#
#    def __init__(
#        self,
#        config: StreetHazardsDatasetConfig,
#        root_dir: Path,
#        coco_dir: Path,
#        subset: str = "training",
#        transforms: Optional[A.Compose] = None,
#    ) -> None:
#        images_dir = root_dir / "images" / subset
#        masks_dir = root_dir / "annotations" / subset
#
#        self.imgs = TorchSerializedList(
#            sorted(str(p) for p in images_dir.rglob("*.png"))
#        )
#        self.masks = TorchSerializedList(
#            sorted(str(p) for p in masks_dir.rglob("*.png"))
#        )
#        self.transforms = transforms
#
#        if len(self.imgs) != len(self.masks):
#            raise ValueError(
#                f"Number of images ({len(self.imgs)}) and masks ({len(self.masks)}) do not match."
#            )
#
#        self.mask_shift = config.mask_shift
#        self.coco_transform = (
#            InsertCOCO(
#                coco_dir=str(coco_dir),
#                exclude_classes="Streethazards",
#                p=1,
#                ood_mask_value=14,
#            )
#            if config.add_anomalies
#            else None
#        )
#
#    def __len__(self) -> int:
#        return len(self.imgs)
#
#    def __getitem__(
#        self, idx: int, apply_transforms: bool = True
#    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
#        img = cv.imread(self.imgs[idx], cv.IMREAD_COLOR_RGB)
#        mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)
#
#        if self.coco_transform is not None:
#            img, mask = self.coco_transform(img, mask)
#            assert isinstance(mask, torch.Tensor), "InsertCOCO returns mask as Tensor"
#            mask = mask.cpu().numpy()
#
#        if apply_transforms and self.transforms is not None:
#            augmented = self.transforms(image=img, mask=mask)
#            img, mask = augmented["image"], augmented["mask"]
#
#        assert img is not None and mask is not None, (
#            "Image or mask is None after transformations"
#        )
#
#        mask = mask - int(self.mask_shift)
#
#        return img, mask
#
#    @property
#    def num_classes(self) -> int:
#        return len(self.CLASSES)
#
#    @property
#    def id_to_label_map(self) -> Dict[int, str]:
#        """
#        Return a mapping from class ID to class name
#        """
#        return {idx: cls_name for idx, cls_name in enumerate(self.CLASSES)}
#
class StreetHazards(Dataset):
    CLASSES = [
        "unlabeled",
        "building",
        "fence",
        "other",
        "pedestrian",
        "pole",
        "road line",
        "road",
        "sidewalk",
        "vegetation",
        "car",
        "wall",
        "traffic sign",
        "anomaly",
    ]

    def __init__(
        self,
        config: StreetHazardsDatasetConfig,
        root_dir: Path,
        coco_dir: Path,
        subset: str = "training",
        transforms: Optional[A.Compose] = None,
    ) -> None:
        images_dir = root_dir / "images" / subset
        masks_dir = root_dir / "annotations" / subset

        self.imgs = TorchSerializedList(
            sorted(str(p) for p in images_dir.rglob("*.png"))
        )
        self.masks = TorchSerializedList(
            sorted(str(p) for p in masks_dir.rglob("*.png"))
        )
        self.transforms = transforms

        if len(self.imgs) != len(self.masks):
            raise ValueError(
                f"Number of images ({len(self.imgs)}) and masks ({len(self.masks)}) do not match."
            )

        self.mask_shift = config.mask_shift
        self.coco_transform = (
            InsertCOCO(
                coco_dir=str(coco_dir),
                exclude_classes="Streethazards",
                p=1,
                ood_mask_value=14,
            )
            if config.add_anomalies
            else None
        )
        self.processor = Mask2FormerImageProcessor()

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

        mask = mask - int(self.mask_shift)

        return img, mask

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

    @property
    def id_to_label_map(self) -> Dict[int, str]:
        """
        Return a mapping from class ID to class name
        """
        return {idx: cls_name for idx, cls_name in enumerate(self.CLASSES)}
