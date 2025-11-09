from pydantic import BeforeValidator

from ml4cv_assignment.config.registry import _instantiate_from_registry


def make_field_before_validator(registry) -> BeforeValidator:
    return BeforeValidator(lambda v: _instantiate_from_registry(v, registry=registry))
