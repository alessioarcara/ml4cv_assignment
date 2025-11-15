from typing import Sequence

import albumentations as A
import numpy as np
import torch


class Denormalize(object):
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        mean_tuple = tuple(-m / s for m, s in zip(mean, std))
        std_tuple = tuple(1.0 / s for s in std)

        self.transform = A.Compose(
            [
                A.Normalize(mean=mean_tuple, std=std_tuple, max_pixel_value=1.0),
                A.FromFloat(max_value=255, dtype="uint8"),
            ]
        )

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
            if img.ndim == 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            else:  # [B, C, H, W]
                img = np.transpose(img.squeeze(), (1, 2, 0))
        return self.transform(image=img)["image"]
