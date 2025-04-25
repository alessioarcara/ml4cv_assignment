import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_data_transforms(
    img_height: int,
    img_width: int,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> dict[str, A.Compose]:
    train_transforms = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            # A.RandomCrop(height=img_height, width=img_width),
            # Basic Geometric
            A.HorizontalFlip(p=0.5),
            # Dropout / Occlusion
            A.OneOf(
                [
                    A.CoarseDropout(
                        p=1.0,
                    ),
                    A.GridDropout(
                        ratio=0.5,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            # Color Augmentations
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.8,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.PlanckianJitter(p=1.0),
                ],
                p=0.7,
            ),
            # Environmental artefacts
            A.OneOf(
                [
                    A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.3, 0.3), p=1.0),
                    A.RandomShadow(p=1.0),
                ],
                p=0.1,
            ),
            # Normalization
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            # A.Normalize(normalization="image_per_channel", p=1.0),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    return {"train": train_transforms, "val": val_transforms}
