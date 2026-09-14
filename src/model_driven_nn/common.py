from collections.abc import Mapping

from .types import ArchitectureSpec


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_mapping(value: object, name: str) -> ArchitectureSpec:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a YAML mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value
