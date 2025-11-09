from typing import Dict, Generic, List, Type, TypeVar, Union

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str, cls: Type[T]) -> None:
        if name in self._registry:
            raise KeyError(f"{name} already registered")
        self._registry[name] = cls

    def instantiate(self, name: str, **kwargs) -> T:
        cls = self._registry.get(name)
        if cls is None:
            raise KeyError(f"{name} not registered")
        try:
            return cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to instantiate {name} with {kwargs}: {e}")


def _instantiate_from_registry(value: Union[dict, List[dict]], registry: Registry):
    if isinstance(value, list):
        return [_instantiate_from_registry(v, registry) for v in value]
    if isinstance(value, dict):
        if "type" in value:
            cls_name = value["type"]
            params = value.get("params", {})
            params = {
                k: _instantiate_from_registry(val, registry)
                for k, val in params.items()
            }
            return registry.instantiate(cls_name, **params)

    return value
