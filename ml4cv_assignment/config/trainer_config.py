import os
from typing import Annotated, List, Optional, Type

import albumentations as A
from pydantic import BaseModel, Field

from ml4cv_assignment.config.registries import (
    callback_registry,
    metric_registry,
    transform_registry,
)
from ml4cv_assignment.config.validator import registry_validator
from ml4cv_assignment.data.transforms import Denormalize
from ml4cv_assignment.training.callbacks import (
    Callback,
)
from ml4cv_assignment.training.metrics import Metric


class TrainerConfig(BaseModel, arbitrary_types_allowed=True):
    train_transforms: Annotated[
        A.Compose,
        registry_validator(transform_registry),
    ] = Field(default_factory=lambda: A.Compose([]))
    val_transforms: Annotated[
        A.Compose,
        registry_validator(transform_registry),
    ] = Field(default_factory=lambda: A.Compose([]))
    metrics: Annotated[List[Metric], registry_validator(metric_registry)] = Field(
        default_factory=list
    )
    callbacks: Annotated[List[Callback], registry_validator(callback_registry)] = Field(
        default_factory=list
    )
    batch_size: int = Field(
        ..., description="Number of samples processed in each training step"
    )
    num_workers: int = Field(
        default_factory=lambda: max((os.cpu_count() or 1) - 1, 0),
        description="Number of DataLoader workers (Defaults to CPU core count minus one, or 0 if unknown)",
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
        description="Device to use for training (e.g., 'cpu', 'cuda'). If None, defaults to automatic selection",
    )
    wandb_project_name: str = Field(..., description="W&B project name for logging")
    wandb_entity: str = Field(..., description="W&B entity (user or team) for logging")
    wandb_base_run_name: str = Field(
        ...,
        description="Base name for the wandb run; a timestamp will be appended",
    )

    def find_transform(
        self, transform_type: Type[A.BasicTransform], in_train: bool
    ) -> Optional[A.BasicTransform]:
        """
        Recursively searches for a specific transform type in the train or validation pipeline.
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

    def get_denormalize(self, in_train: bool = False) -> Denormalize:
        """
        Factories a Denormalize object based on the Normalize transform found in the specified pipeline.
        """
        normalize_transform: Optional[A.Normalize] = self.find_transform(
            A.Normalize, in_train=in_train
        )
        if normalize_transform is None:
            pipeline_name = "train" if in_train else "validation"
            raise ValueError(
                f"No Normalize transform found in the {pipeline_name} pipeline."
            )

        return Denormalize(mean=normalize_transform.mean, std=normalize_transform.std)
