from typing import Annotated

import torch.nn as nn
from pydantic import BaseModel, Field

from ml4cv_assignment.config.registries import loss_registry, model_registry
from ml4cv_assignment.config.validator import registry_validator


class ModelConfig(BaseModel, arbitrary_types_allowed=True):
    model: Annotated[nn.Module, registry_validator(model_registry, loss_registry)] = (
        Field(..., description="The model to train")
    )
    use_preprocessor: bool = Field(
        False, description="Whether to use the Mask2Former processor during collation"
    )
