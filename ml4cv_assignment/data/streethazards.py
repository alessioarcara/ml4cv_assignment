from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import numpy as np
from PIL import Image
from pytorch_ood.augment import InsertCOCO
from torch import Tensor
from torch.utils.data import Dataset

from ml4cv_assignment.config.dataset_config import StreetHazardsDatasetConfig
from ml4cv_assignment.data.data_utils import TorchSerializedList


class StreetHazards(Dataset):
    CLASSES = [
        "unlabeled",  # 0
        "building",  # 1
        "fence",  # 2
        "other",  # 3
        "pedestrian",  # 4
        "pole",  # 5
        "road line",  # 6
        "road",  # 7
        "sidewalk",  # 8
        "vegetation",  # 9
        "car",  # 10
        "wall",  # 11
        "traffic sign",  # 12
        "anomaly",  # 13
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
                p=0.1,
                n=config.num_objects_to_insert,
                ood_mask_value=14,
            )
            if config.add_anomalies
            else None
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(
        self, idx: int, apply_transforms: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        if self.coco_transform is not None:
            img, mask = self.coco_transform(img, mask)

        img_np = np.array(img)
        mask_np = np.array(mask)

        if apply_transforms and self.transforms is not None:
            augmented = self.transforms(image=img_np, mask=mask_np)
            img_np, mask_np = augmented["image"], augmented["mask"]

        assert img_np is not None and mask_np is not None, (
            "Image or mask is None after transformations"
        )

        mask_np = mask_np - int(self.mask_shift)

        return img_np, mask_np

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

    @classmethod
    def id_to_label_map(cls) -> Dict[int, str]:
        """
        Return a mapping from class ID to class name
        """
        return {idx: cls_name for idx, cls_name in enumerate(cls.CLASSES)}
