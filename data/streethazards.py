import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


STREET_HAZARDS_CLASSES = [
    "unlabeled", "building", "fence", "other", "pedestrian",
    "pole", "road line", "road", "sidewalk", "vegetation",
    "car", "wall", "traffic sign", "anomaly"
]


def get_classes_as_dict():
        return dict(enumerate(STREET_HAZARDS_CLASSES))


# Implementazione basata su Detectron2:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
class TorchSerializedList:
    """
    A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works.
    """

    def __init__(self, lst: list):
        self._lst = lst

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        self._lst = [_serialize(x) for x in self._lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(self._lst))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


class StreetHazards(Dataset):
    def __init__(
            self, 
            root_dir: Path, 
            subset: str = "training", 
            transforms=None
        ) -> None:
        images_dir = root_dir / "images" / subset
        masks_dir = root_dir / "annotations" / subset

        self.imgs = TorchSerializedList(sorted(str(p) for p in images_dir.rglob("*.png")))
        self.masks = TorchSerializedList(sorted(str(p) for p in masks_dir.rglob("*.png")))
        self.transforms = transforms 

        if len(self.imgs) - len(self.masks) != 0:
            raise AssertionError(
                f"Labels and Images differ in size {len(self.imgs) - len(self.masks)}."
            )

    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def _load_image(path: str):
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img)

    @staticmethod
    def _load_mask(path: str):
        with Image.open(path) as mask:
            mask = mask.convert("L")
            return np.array(mask)

    def __getitem__(self, idx):
        path_img = self.imgs[idx]
        path_mask = self.masks[idx]
        img = self._load_image(path_img)
        mask = self._load_mask(path_mask)

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, (mask - 1)
    
    def get_class_weights(self) -> torch.Tensor:
        num_classes = len(STREET_HAZARDS_CLASSES)
        class_counts = torch.zeros(num_classes)
        for path_mask in self.masks:
            mask = torch.from_numpy(self._load_mask(path_mask))
            unique_labels = torch.unique(mask).long() - 1
            class_counts[unique_labels] += 1
        return class_counts
    

if __name__ == "__main__":
    current_dir = Path(__file__).parent  # Directory dello script
    dataset_path = current_dir / "datasets/train"

    dataset = StreetHazards(
        root_dir=dataset_path,
        subset="training/t1-3/"
    )

    img, mask = dataset[0]
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)