from collections.abc import Mapping
from pathlib import Path

import yaml

from .builder import ArchitectureSpec, Builder
from .components import Sequential
from .network import Network


class Loader:
    def __init__(self, builder: Builder | None = None) -> None:
        self.builder = builder if builder is not None else Builder()

    def load(self, path: str | Path) -> Network:
        with Path(path).open(encoding="utf-8") as file:
            spec = yaml.safe_load(file)

        architecture = self._validate_mapping(spec, "Architecture")
        name = architecture.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Architecture name must be a nonblank string")

        input_dim = self._positive_int(
            architecture.get("input_dim"), "Architecture input_dim"
        )
        expected_output = None
        if "output_dim" in architecture:
            expected_output = self._positive_int(
                architecture["output_dim"], "Architecture output_dim"
            )

        model_spec = self._validate_mapping(
            architecture.get("model"), "Architecture model"
        )
        if model_spec.get("type") != "sequential":
            raise ValueError("Architecture model type must be 'sequential'")

        model, actual_output_dim = self.builder.build(model_spec, input_dim)
        if not isinstance(model, Sequential):
            raise TypeError("Architecture model must build a Sequential component")
        if expected_output is not None and actual_output_dim != expected_output:
            raise ValueError(
                f"Architecture declares output_dim={expected_output}, "
                f"but builder inferred {actual_output_dim} features"
            )

        return Network(name, model, input_dim, actual_output_dim)

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _validate_mapping(value: object, name: str) -> ArchitectureSpec:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a YAML mapping")
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        return value
