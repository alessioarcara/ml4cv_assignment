from typing import Optional

from pydantic import BaseModel, DirectoryPath, Field, FilePath


class PathsConfig(BaseModel):
    street_hazards_train_dir: DirectoryPath = Field(
        ...,
        description="Directory containing the StreetHazards training dataset",
    )
    street_hazards_test_dir: DirectoryPath = Field(
        ...,
        description="Directory containing the StreetHazards test dataset",
    )
    checkpoint: Optional[FilePath] = Field(
        None,
        description="Directory where model checkpoints are stored",
    )
    coco_data_dir: DirectoryPath = Field(
        ..., description="Directory containing the COCO dataset"
    )
