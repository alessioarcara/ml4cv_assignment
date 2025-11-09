import albumentations as A
import torch.nn as nn
from kornia.losses import FocalLoss

from ml4cv_assignment.config.config import TrainerConfig
from ml4cv_assignment.training.losses import WeightedLoss

losses_cfg = [
    {
        "type": "WeightedLoss",
        "params": {
            "loss": {
                "type": "FocalLoss",
                "params": {"alpha": 0.25, "gamma": 2.0},
            },
            "weight": 2.0,
        },
    },
    {"type": "CrossEntropyLoss", "params": {}},
]
transforms_cfg = {
    "type": "Compose",
    "params": {
        "transforms": [
            {"type": "Resize", "params": {"height": 720, "width": 1280}},
            {
                "type": "OneOf",
                "params": {
                    "transforms": [
                        {"type": "RandomSunFlare", "params": {"p": 1.0}},
                        {"type": "RandomRain", "params": {"p": 1.0}},
                    ],
                    "p": 1.0,
                },
            },
            {
                "type": "Normalize",
                "params": {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "max_pixel_value": 255.0,
                },
            },
            {"type": "ToTensorV2", "params": {}},
        ]
    },
}
# {"metrics": [{"type": "IoU"}]}


def test_losses_config():
    cfg = TrainerConfig(losses=losses_cfg)

    assert isinstance(cfg.losses[0], WeightedLoss)
    assert isinstance(cfg.losses[1], nn.CrossEntropyLoss)

    assert cfg.losses[0].weight == 2.0
    focal_loss = cfg.losses[0].loss
    assert isinstance(focal_loss, FocalLoss)
    assert focal_loss.alpha == 0.25
    assert focal_loss.gamma == 2.0


def test_transforms_config():
    cfg = TrainerConfig(train_transforms=transforms_cfg)

    train_compose = cfg.train_transforms

    assert isinstance(train_compose, A.Compose)

    assert any(isinstance(t, A.Resize) for t in train_compose.transforms)
    assert any(isinstance(t, A.Normalize) for t in train_compose.transforms)
    assert any(isinstance(t, A.ToTensorV2) for t in train_compose.transforms)


def test_metrics_config():
    assert False
