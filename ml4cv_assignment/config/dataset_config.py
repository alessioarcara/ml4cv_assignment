from abc import ABC, abstractmethod
from typing import Literal, Optional

import albumentations as A
from pydantic import BaseModel, Field

from ml4cv_assignment.config.paths_config import PathsConfig
from ml4cv_assignment.data.cityscapes import CityscapesDataset
from ml4cv_assignment.data.streethazards import StreetHazards
from ml4cv_assignment.utils.typings import SplitType


class BaseDatasetConfig(BaseModel, ABC):
    add_anomalies: bool = Field(..., description="Whether to add OOD anomalies")
    num_objects_to_insert: int = Field(
        1, description="Number of synthetic OOD objects to insert into the scene", ge=1
    )
    prob_insert: float = Field(
        1.0,
        description="Probability of inserting synthetic OOD objects into each image",
        ge=0.0,
        le=1.0,
    )

    @abstractmethod
    def create_dataset(
        self,
        paths: PathsConfig,
        subset: SplitType,
        transforms: Optional[A.Compose] = None,
    ):
        pass


class StreetHazardsDatasetConfig(BaseDatasetConfig):
    type: Literal["streethazards"] = "streethazards"

    mask_shift: bool = Field(
        ..., description="Whether to shift mask labels (1-based -> 0-based)"
    )

    def create_dataset(
        self,
        paths: PathsConfig,
        subset: SplitType,
        transforms: Optional[A.Compose] = None,
    ):
        if subset == SplitType.TEST:
            dataset_root = paths.street_hazards_test_dir
        else:
            dataset_root = paths.street_hazards_train_dir

        return StreetHazards(
            config=self,
            root_dir=dataset_root,
            coco_dir=paths.coco_data_dir,
            transforms=transforms,
            subset=subset,
        )


class CityscapesDatasetConfig(BaseDatasetConfig):
    type: Literal["cityscapes"] = "cityscapes"

    def create_dataset(
        self,
        paths: PathsConfig,
        subset: SplitType,
        transforms: Optional[A.Compose] = None,
    ):
        split_mapping: dict[SplitType, str] = {
            SplitType.TRAINING: "train",
            SplitType.VALIDATION: "val",
            SplitType.TEST: "test",
        }

        cityscapes_split = split_mapping.get(subset)

        if not cityscapes_split:
            raise ValueError(f"Invalid subset '{subset}' for Cityscapes dataset")

        return CityscapesDataset(
            root=paths.cityscapes_root_dir,
            split=cityscapes_split,
            mode="fine",
            target_type="semantic",
            transform=transforms,
        )
