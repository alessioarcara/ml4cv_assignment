from typing import Annotated, Dict, Optional

from pydantic import BaseModel, Field

from ml4cv_assignment.config.registries import loss_registry, model_registry
from ml4cv_assignment.config.validator import registry_validator
from ml4cv_assignment.models.base_model import BaseModel as MyModel


class PreprocessorConfig(BaseModel):
    ignore_index: int = Field(
        -100,
        description="The value to ignore in the segmentation masks during preprocessing",
    )


class ModelConfig(BaseModel, arbitrary_types_allowed=True):
    model: Annotated[MyModel, registry_validator(model_registry, loss_registry)] = (
        Field(..., description="The model to train")
    )
    preprocessor: Optional[PreprocessorConfig] = Field(
        None, description="Configuration for the Mask2Former preprocessor"
    )
    ckpt_remap: Dict[str, str] = Field(
        default_factory=dict,
        description="Map old checkpoint keys to new keys when loading weights",
    )
