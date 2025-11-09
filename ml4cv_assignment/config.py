from pydantic import BaseModel, ConfigDict, DirectoryPath, Field

from ml4cv_assignment.data.streethazards import StreetHazards
from ml4cv_assignment.utils.io import read_yaml
from ml4cv_assignment.utils.typings import PathOrStr


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


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")
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

    @classmethod
    def load(cls, path: PathOrStr) -> "Config":
        """Load configuration from a YAML file"""
        data = read_yaml(path)
        return cls(**data)
