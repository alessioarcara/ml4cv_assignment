from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image
from pytorch_ood.augment import InsertCOCO

if TYPE_CHECKING:
    from ml4cv_assignment.config.dataset_config import BaseDatasetConfig


class AnomalyAugmentationMixin:
    def setup_coco(
        self,
        config: "BaseDatasetConfig",
        coco_dir: Union[Path, str],
        exclude_classes: Optional[list[str] | str] = None,
    ):
        # HACK: If exclude_classes is None or empty, InsertCOCO internally filters out everything.
        # We provide a non-existent class to force it to include all images.
        if not exclude_classes:
            exclude_classes = ["__fake_class_to_bypass_bug__"]

        self.coco_transform = (
            InsertCOCO(
                coco_dir=str(coco_dir),
                exclude_classes=exclude_classes,
                p=config.prob_insert,
                n=config.num_objects_to_insert,
                ood_mask_value=config.unknown_mask_value,
            )
            if config.add_anomalies
            else None
        )

    def apply_augmentations(
        self,
        img: Image.Image,
        mask_np: np.ndarray,
        transforms: Optional[Callable],
        apply_transforms: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Apply COCO Anomaly Insertion
        if hasattr(self, "coco_transform") and self.coco_transform is not None:
            img, mask_tensor = self.coco_transform(img, mask_np)
            # InsertCOCO returns mask as a Tensor; convert back to NumPy (uint8)
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

        # 2. Convert to NumPy (Needed for Albumentations)
        img_np = np.array(img)

        # 3. Apply Albumentations (NumPy -> NumPy)
        if apply_transforms and transforms is not None:
            augmented = transforms(image=img_np, mask=mask_np)
            img_np, mask_np = augmented["image"], augmented["mask"]

        return img_np, mask_np
