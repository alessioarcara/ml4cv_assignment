from pydantic import BeforeValidator

from ml4cv_assignment.config.registry import _instantiate_from_registry


def registry_instantiation_validator(registry) -> BeforeValidator:
    """
    Returns a Pydantic BeforeValidator that instantiates objects from the given registry
    """
    return BeforeValidator(lambda v: _instantiate_from_registry(v, registry=registry))
