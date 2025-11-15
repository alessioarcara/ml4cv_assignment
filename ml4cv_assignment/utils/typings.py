from enum import StrEnum
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, TypeAlias, Union

PathOrStr: TypeAlias = Union[Path, str]
ClassIdCounter: TypeAlias = CounterType[int]
StepOutput: TypeAlias = Dict[str, float]
MetricResults: TypeAlias = Dict[str, float]


class Stage(StrEnum):
    TRAIN = "train"
    VAL = "val"
