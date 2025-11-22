from typing import Literal

from pydantic import BaseModel, Field


class StreetHazardsDatasetConfig(BaseModel):
    type: Literal["streethazards"]
    add_anomalies: bool = Field(..., description="Whether to add OOD anomalies")
    mask_shift: bool = Field(
        ..., description="Whether to shift mask labels (1-based -> 0-based)"
    )
    num_objects_to_insert: int = Field(
        1, description="Number of synthetic OOD objects to insert into the scene", ge=1
    )
