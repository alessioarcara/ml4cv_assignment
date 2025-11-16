from typing import Annotated, List

import torch.nn as nn
from pydantic import BaseModel, Field

from ml4cv_assignment.config.registries import loss_registry, model_registry
from ml4cv_assignment.config.validator import registry_instantiation_validator


class ModelConfig(BaseModel, arbitrary_types_allowed=True):
    losses: Annotated[
        List[nn.Module], registry_instantiation_validator(loss_registry)
    ] = Field(
        default_factory=list,
        description="Loss functions to use during training",
    )
    model: Annotated[nn.Module, registry_instantiation_validator(model_registry)] = (
        Field(..., description="The model to train")
    )
