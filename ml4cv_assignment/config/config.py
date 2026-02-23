from functools import partial
from typing import Annotated, Callable, Optional, Union

import albumentations as A
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
from transformers import Mask2FormerImageProcessor

from ml4cv_assignment.config.dataset_config import (
    CityscapesDatasetConfig,
    StreetHazardsDatasetConfig,
)
from ml4cv_assignment.config.model_config import ModelConfig
from ml4cv_assignment.config.paths_config import PathsConfig
from ml4cv_assignment.config.trainer_config import TrainerConfig
from ml4cv_assignment.data.collate import collate_fn
from ml4cv_assignment.data.data_utils import MultiEpochsDataLoader
from ml4cv_assignment.models.base_model import BaseModel as MyModel
from ml4cv_assignment.utils.checkpoint import load_checkpoint
from ml4cv_assignment.utils.typings import SplitType


class Config(BaseModel):
    seed: int = Field(..., description="Random seed")
    training: TrainerConfig
    paths: PathsConfig
    dataset_config: Annotated[
        Union[StreetHazardsDatasetConfig, CityscapesDatasetConfig],
        Field(discriminator="type"),
    ]
    model_cfg: ModelConfig

    def dataset(
        self,
        subset: SplitType,
        transforms: Optional[A.Compose] = None,
    ) -> Dataset:
        return self.dataset_config.create_dataset(
            paths=self.paths,
            subset=subset,
            transforms=transforms,
        )

    @property
    def train_dataset(self) -> Dataset:
        return self.dataset(
            subset=SplitType.TRAINING,
            transforms=self.training.train_transforms,
        )

    @property
    def val_dataset(self) -> Dataset:
        return self.dataset(
            subset=SplitType.VALIDATION,
            transforms=self.training.val_transforms,
        )

    @property
    def test_dataset(self) -> Dataset:
        return self.dataset(
            subset=SplitType.TEST,
            transforms=self.training.val_transforms,
        )

    @property
    def dataloader(self) -> Callable[..., DataLoader]:
        processor = None
        if self.model_cfg.preprocessor:
            processor = Mask2FormerImageProcessor(
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
                num_labels=13,
                ignore_index=self.model_cfg.preprocessor.ignore_index,
            )

        collate_func = partial(collate_fn, processor=processor)

        return partial(
            MultiEpochsDataLoader,
            batch_size=self.training.batch_size,
            num_workers=self.training.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_func,
        )

    @property
    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset, shuffle=False)

    @property
    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset, shuffle=False)

    @property
    def model(self) -> MyModel:
        model = self.model_cfg.model

        if self.paths.checkpoint:
            load_checkpoint(model, self.paths.checkpoint, self.model_cfg.ckpt_remap)

        return model
