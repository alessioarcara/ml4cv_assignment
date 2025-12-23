from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from torchvision.datasets import Cityscapes


class CityscapesDataset(Cityscapes):
    def __init__(
        self,
        root: Union[str, Path],
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

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        img_np = np.array(image)
        mask_np = np.array(mask)

        mask_np = self.id_to_train_id[mask_np]

        if self.albu_transforms is not None:
            augmented = self.albu_transforms(image=img_np, mask=mask_np)
            img_np, mask_np = augmented["image"], augmented["mask"]

        return img_np, mask_np

    @classmethod
    def id_to_label_map(cls) -> dict[int, str]:
        id2label = {}
        for c in cls.classes:
            if c.train_id not in [255, -1]:
                id2label[c.train_id] = c.name
        return id2label
