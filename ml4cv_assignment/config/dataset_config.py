from typing import Literal

from pydantic import BaseModel, Field


class StreetHazardsDatasetConfig(BaseModel):
    type: Literal["streethazards"]
    mask_shift: bool = Field(
        ..., description="Whether to shift mask labels (1-based -> 0-based)"
    )
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
