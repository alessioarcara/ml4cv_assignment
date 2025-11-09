from typing import Annotated, List

import albumentations as A
import torch.nn as nn
from kornia.losses import FocalLoss
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from ml4cv_assignment.config.registry import Registry
from ml4cv_assignment.config.validator import make_field_before_validator
from ml4cv_assignment.data.data_utils import MultiEpochsDataLoader
from ml4cv_assignment.data.streethazards import StreetHazards
from ml4cv_assignment.training.losses import (
    ObjectosphereLoss,
    OWLoss,
    PrototypicalGlobalLocalTripletLoss,
    WeightedLoss,
)
from ml4cv_assignment.utils.io import read_yaml
from ml4cv_assignment.utils.typings import PathOrStr

# ------------------------
# Registry losses
# ------------------------
loss_registry = Registry[nn.Module]()
loss_registry.register("FocalLoss", FocalLoss)
loss_registry.register("WeightedLoss", WeightedLoss)
loss_registry.register("CrossEntropyLoss", CrossEntropyLoss)
loss_registry.register("OWLoss", OWLoss)
loss_registry.register(
    "PrototypicalGlobalLocalTripletLoss", PrototypicalGlobalLocalTripletLoss
)
loss_registry.register("ObjectosphereLoss", ObjectosphereLoss)

# ------------------------
# Registry transformations
# ------------------------
transforms_registry = Registry[A.BasicTransform]()
# --- Resize & geometric
transforms_registry.register("Resize", A.Resize)
transforms_registry.register("HorizontalFlip", A.HorizontalFlip)
# --- Color & brightness ---
transforms_registry.register("ColorJitter", A.ColorJitter)
transforms_registry.register("RGBShift", A.RGBShift)
# --- Erasing ---
transforms_registry.register("CoarseDropout", A.CoarseDropout)
# --- Environmental artefacts ---
transforms_registry.register("RandomSunFlare", A.RandomSunFlare)
transforms_registry.register("RandomShadow", A.RandomShadow)
transforms_registry.register("RandomFog", A.RandomFog)
transforms_registry.register("RandomRain", A.RandomRain)
transforms_registry.register("Normalize", A.Normalize)
transforms_registry.register("ToTensorV2", A.ToTensorV2)
transforms_registry.register("OneOf", A.OneOf)
transforms_registry.register("Compose", A.Compose)


class PathsConfig(BaseModel):
    street_hazards_train_dir: DirectoryPath = Field(
        ...,
        description="Directory containing the StreetHazards training dataset",
    )
    street_hazards_test_dir: DirectoryPath = Field(
        ...,
        description="Directory containing the StreetHazards test dataset",
    )
    checkpoints_dir: DirectoryPath = Field(
        ...,
        description="Directory where model checkpoints are stored",
    )
    coco_data_dir: DirectoryPath = Field(
        ..., description="Directory containing the COCO dataset"
    )


class TrainerConfig(BaseModel, arbitrary_types_allowed=True):
    losses: Annotated[List[nn.Module], make_field_before_validator(loss_registry)] = (
        Field(default_factory=list)
    )
    train_transforms: Annotated[
        A.Compose,
        make_field_before_validator(transforms_registry),
    ] = Field(default_factory=lambda: A.Compose([]))
    val_transforms: Annotated[
        A.Compose,
        make_field_before_validator(transforms_registry),
    ] = Field(default_factory=lambda: A.Compose([]))


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")
    training: TrainerConfig
    paths: PathsConfig

    @property
    def train_dataset(self) -> StreetHazards:
        return StreetHazards(
            root_dir=self.paths.street_hazards_train_dir, subset="training"
        )

    @property
    def val_dataset(self) -> StreetHazards:
        return StreetHazards(
            root_dir=self.paths.street_hazards_train_dir,
            subset="validation",
        )

    @property
    def train_dataloader(self) -> DataLoader:
        return MultiEpochsDataLoader(
            self.train_dataset,
            shuffle=True,
        )

    @property
    def val_dataloader(self) -> DataLoader:
        return MultiEpochsDataLoader(
            self.val_dataset,
            shuffle=False,
        )

    @classmethod
    def load(cls, path: PathOrStr) -> "Config":
        """Load configuration from a YAML file"""
        data = read_yaml(path)
        return cls(**data)
