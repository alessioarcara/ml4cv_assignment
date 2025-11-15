import os
from typing import Annotated, List, Optional, Type

import albumentations as A
import torch.nn as nn
from pydantic import BaseModel, Field

from ml4cv_assignment.config.registries import (
    callback_registry,
    loss_registry,
    metric_registry,
    transform_registry,
)
from ml4cv_assignment.config.validator import registry_instantiation_validator
from ml4cv_assignment.data.transforms import Denormalize
from ml4cv_assignment.training.callbacks import (
    Callback,
)
from ml4cv_assignment.training.metrics import Metric


class TrainerConfig(BaseModel, arbitrary_types_allowed=True):
    losses: Annotated[
        List[nn.Module], registry_instantiation_validator(loss_registry)
    ] = Field(default_factory=list)
    train_transforms: Annotated[
        A.Compose,
        registry_instantiation_validator(transform_registry),
    ] = Field(default_factory=lambda: A.Compose([]))
    val_transforms: Annotated[
        A.Compose,
        registry_instantiation_validator(transform_registry),
    ] = Field(default_factory=lambda: A.Compose([]))
    metrics: Annotated[
        List[Metric], registry_instantiation_validator(metric_registry)
    ] = Field(default_factory=list)
    callbacks: Annotated[
        List[Callback], registry_instantiation_validator(callback_registry)
    ] = Field(default_factory=list)
    batch_size: int = Field(
        ..., description="Number of samples processed in each training step"
    )
    num_workers: int = Field(
        default_factory=lambda: max((os.cpu_count() or 1) - 1, 0),
        description="Number of DataLoader workers (Defaults to CPU core count minus one, or 0 if unknown)",
    )
    lr: float = Field(..., description="Learning rate used by the optimizer")
    weight_decay: float = Field(
        ..., description="L2 regularization factor applied to the model weights"
    )
    num_epochs: int = Field(
        ..., description="Total number of training epochs over the dataset"
    )
    use_mixed_precision: bool = Field(
        ..., description="Whether to enable mixed precision training"
    )
    evaluation_rate: int = Field(
        ..., description="Frequency (in epochs) at which the model is evaluated"
    )
    use_torch_compile: bool = Field(
        ...,
        description="Whether to compile the model using torch.compile",
    )
    device: Optional[str] = Field(
        default=None,
        description="Device to use for training (e.g., 'cpu', 'cuda'). If None, defaults to automatic selection.",
    )
    wandb_project_name: str = Field(..., description="W&B project name for logging")
    wandb_entity: str = Field(..., description="W&B entity (user or team) for logging")

    def find_transform(
        self, transform_type: Type[A.BasicTransform], in_train: bool
    ) -> Optional[A.BasicTransform]:
        """
        Search for a transform in train or validation pipeline
        """
        transforms = (
            self.train_transforms.transforms
            if in_train
            else self.val_transforms.transforms
        )

        def search(transforms_list: list) -> Optional[A.BasicTransform]:
            for t in transforms_list:
                if isinstance(t, transform_type):
                    return t
                if isinstance(t, (A.Compose, A.OneOf)):
                    result = search(t.transforms)
                    if result is not None:
                        return result
            return None

        return search(transforms)

    @property
    def denormalize(self) -> Denormalize:
        normalize: Optional[A.Normalize] = self.find_transform(
            A.Normalize, in_train=False
        )
        if normalize is None:
            raise ValueError("No Normalize transform found in val_transforms")

        return Denormalize(mean=normalize.mean, std=normalize.std)
