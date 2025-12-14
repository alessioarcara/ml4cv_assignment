from functools import partial
from pathlib import Path
from typing import Annotated, Callable, Optional, Union

import albumentations as A
import torch
from loguru import logger
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from transformers import Mask2FormerImageProcessor

from ml4cv_assignment.config.dataset_config import StreetHazardsDatasetConfig
from ml4cv_assignment.config.model_config import ModelConfig
from ml4cv_assignment.config.paths_config import PathsConfig
from ml4cv_assignment.config.trainer_config import TrainerConfig
from ml4cv_assignment.data.collate import collate_fn
from ml4cv_assignment.data.data_utils import MultiEpochsDataLoader
from ml4cv_assignment.data.streethazards import StreetHazards
from ml4cv_assignment.models.base_model import BaseModel as MyModel
from ml4cv_assignment.utils.checkpoint import remap_state_dict_keys


class Config(BaseModel):
    seed: int = Field(..., description="Random seed")
    training: TrainerConfig
    paths: PathsConfig
    dataset_config: Annotated[
        Union[StreetHazardsDatasetConfig], Field(discriminator="type")
    ]
    model_cfg: ModelConfig

    def dataset(
        self,
        root_dir: Path,
        transforms: Optional[A.Compose] = None,
        subset: str = "training",
    ) -> "StreetHazards":
        return StreetHazards(
            config=self.dataset_config,
            root_dir=root_dir,
            coco_dir=self.paths.coco_data_dir,
            transforms=transforms,
            subset=subset,
        )

    @property
    def train_dataset(self) -> "StreetHazards":
        return self.dataset(
            root_dir=self.paths.street_hazards_train_dir,
            transforms=self.training.train_transforms,
            subset="training",
        )

    @property
    def val_dataset(self) -> "StreetHazards":
        return self.dataset(
            root_dir=self.paths.street_hazards_train_dir,
            transforms=self.training.val_transforms,
            subset="validation",
        )

    @property
    def test_dataset(self) -> "StreetHazards":
        return self.dataset(
            root_dir=self.paths.street_hazards_test_dir,
            transforms=self.training.val_transforms,
            subset="test",
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
            ckpt = torch.load(self.paths.checkpoint, map_location="cpu")

            state_dict = remap_state_dict_keys(ckpt, self.model_cfg.ckpt_remap)

            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )

            if len(missing_keys) > 0:
                logger.warning(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

            logger.info("✅ Checkpoint loaded successfully")

        return model
