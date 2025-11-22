from enum import StrEnum
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, TypeAlias, Union

from torch import Tensor

PathOrStr: TypeAlias = Union[Path, str]
ClassIdCounter: TypeAlias = CounterType[int]
StepOutput: TypeAlias = Dict[str, float]
MetricResults: TypeAlias = Dict[str, float]
Batch: TypeAlias = Dict[str, Tensor]


class Stage(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
