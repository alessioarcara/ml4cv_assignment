from functools import partial
from typing import Annotated, Callable, Union

import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from transformers import Mask2FormerImageProcessor

from ml4cv_assignment.config.dataset_config import StreetHazardsDatasetConfig
from ml4cv_assignment.config.paths_config import PathsConfig

# from ml4cv_assignment.config.registries import model_registry
from ml4cv_assignment.config.trainer_config import TrainerConfig

# from ml4cv_assignment.config.validator import registry_instantiation_validator
from ml4cv_assignment.data.data_utils import MultiEpochsDataLoader
from ml4cv_assignment.data.streethazards import StreetHazards
from ml4cv_assignment.utils.io import read_yaml
from ml4cv_assignment.utils.typings import PathOrStr


def collate_fn(batch, processor):
    images, masks = zip(*batch)
    inputs = processor(
        images=list(images), segmentation_maps=list(masks), return_tensors="pt"
    )
    inputs["orig_masks"] = torch.stack(masks)
    return inputs


class Config(BaseModel):
    seed: int = Field(..., description="Random seed")
    training: TrainerConfig
    paths: PathsConfig
    dataset_config: Annotated[
        Union[StreetHazardsDatasetConfig], Field(discriminator="type")
    ]

    @property
    def train_dataset(self) -> "StreetHazards":
        return StreetHazards(
            config=self.dataset_config,
            root_dir=self.paths.street_hazards_train_dir,
            coco_dir=self.paths.coco_data_dir,
            transforms=self.training.train_transforms,
            subset="training",
        )

    @property
    def val_dataset(self) -> "StreetHazards":
        return StreetHazards(
            config=self.dataset_config,
            root_dir=self.paths.street_hazards_train_dir,
            coco_dir=self.paths.coco_data_dir,
            transforms=self.training.val_transforms,
            subset="validation",
        )

    @property
    def dataloader(self) -> Callable[..., DataLoader]:
        processor = Mask2FormerImageProcessor(
            do_resize=False,
            do_normalize=False,
            num_labels=13,
            ignore_index=255,
        )
        collate_with_processor = partial(collate_fn, processor=processor)

        return partial(
            MultiEpochsDataLoader,
            batch_size=self.training.batch_size,
            num_workers=self.training.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_with_processor,
        )

    @property
    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset, shuffle=False)

    @classmethod
    def load(cls, path: PathOrStr) -> "Config":
        """Load configuration from a YAML file"""
        data = read_yaml(path)
        return cls.model_validate(data)
