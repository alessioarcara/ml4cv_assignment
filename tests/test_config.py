from typing import Any, Dict

import albumentations as A
import pytest

from ml4cv_assignment.config.config import TrainerConfig
from ml4cv_assignment.training.callbacks import (
    EarlyStoppingCallback,
    ModelSavingCallback,
)
from ml4cv_assignment.training.metrics import AUPR, MeanIoU

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
metrics_cfg = [
    {"type": "MeanIoU", "params": {"num_classes": 13}},
    {"type": "AUPR", "params": {"unknown_label": 255}},
]
callbacks_cfg = [
    {
        "type": "EarlyStoppingCallback",
        "params": {"patience": 5, "history_key": "val-mIoU", "minimize": False},
    },
    {
        "type": "ModelSavingCallback",
        "params": {
            "out_dir": "./checkpoints",
            "history_key": "val-mIoU",
            "minimize": False,
        },
    },
]


@pytest.fixture
def base_trainer_kwargs():
    return dict(
        batch_size=1,
        num_epochs=1,
        use_mixed_precision=False,
        evaluation_rate=1,
        use_torch_compile=False,
        wandb_project_name="test_project",
        wandb_entity="test_entity",
        wandb_base_run_name="test_run",
    )


def test_transforms_config(base_trainer_kwargs: Dict[str, Any]):
    cfg = TrainerConfig(**base_trainer_kwargs, train_transforms=transforms_cfg)

    train_compose = cfg.train_transforms

    assert isinstance(train_compose, A.Compose)

    assert any(isinstance(t, A.Resize) for t in train_compose.transforms)
    assert any(isinstance(t, A.Normalize) for t in train_compose.transforms)
    assert any(isinstance(t, A.ToTensorV2) for t in train_compose.transforms)


def test_metrics_config(base_trainer_kwargs: Dict[str, Any]):
    cfg = TrainerConfig(**base_trainer_kwargs, metrics=metrics_cfg)  # type: ignore

    assert isinstance(cfg.metrics[0], MeanIoU)
    assert isinstance(cfg.metrics[1], AUPR)


def test_callbacks_config(base_trainer_kwargs: Dict[str, Any]):
    cfg = TrainerConfig(**base_trainer_kwargs, callbacks=callbacks_cfg)  # type: ignore

    assert len(cfg.callbacks) == 2
    assert isinstance(cfg.callbacks[0], EarlyStoppingCallback)
    assert isinstance(cfg.callbacks[1], ModelSavingCallback)
