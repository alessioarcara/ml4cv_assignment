from pathlib import Path
from typing import Counter as CounterType
from typing import TypeAlias, Union

PathOrStr: TypeAlias = Union[Path, str]
ClassIdCounter: TypeAlias = CounterType[int]
