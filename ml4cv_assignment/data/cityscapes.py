from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np
from torch import Tensor
from torchvision.datasets import Cityscapes

from ml4cv_assignment.data.dataset_mixins import AnomalyAugmentationMixin

if TYPE_CHECKING:
    from ml4cv_assignment.config.dataset_config import CityscapesDatasetConfig


class CityscapesDataset(Cityscapes, AnomalyAugmentationMixin):
    def __init__(
        self,
        config: "CityscapesDatasetConfig",
        root: Union[str, Path],
        coco_dir: Path,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[list[str], str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transform=None,
            target_transform=None,
        )
        self.albu_transforms = transform

        # lookup table to map original IDs to training IDs
        self.id_to_train_id = np.full((256,), 255, dtype=np.uint8)

        for cls in self.classes:
            if cls.id >= 0:  # ignore classes with id -1
                self.id_to_train_id[cls.id] = cls.train_id

        self.setup_coco(config, coco_dir=coco_dir)

    def __getitem__(
        self, idx: int, apply_transforms: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        img, mask = super().__getitem__(idx)

        mask_np = np.array(mask)
        # Map original IDs to training IDs
        mask_mapped = self.id_to_train_id[mask_np]

        img_aug, mask_aug = self.apply_augmentations(
            img, mask_mapped, self.albu_transforms, apply_transforms
        )

        return img_aug, mask_aug

    @classmethod
    def id_to_label_map(cls) -> dict[int, str]:
        id2label = {}
        for c in cls.classes:
            if c.train_id not in [255, -1]:
                id2label[c.train_id] = c.name
        id2label[19] = "anomaly"
        return id2label
