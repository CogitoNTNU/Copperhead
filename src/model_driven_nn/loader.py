from pathlib import Path

import yaml

from .builder import Builder
from .common import positive_int, validate_mapping
from .components import Structure
from .network import Network


class Loader:
    def __init__(self, builder: Builder | None = None) -> None:
        self.builder = builder if builder is not None else Builder()

    def load(self, path: str | Path) -> Network:
        with Path(path).open(encoding="utf-8") as file:
            spec = yaml.safe_load(file)

        architecture = validate_mapping(spec, "Architecture")
        name = architecture.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Architecture name must be a nonblank string")

        input_dim = positive_int(
            architecture.get("input_dim"), "Architecture input_dim"
        )
        expected_output = None
        if "output_dim" in architecture:
            expected_output = positive_int(
                architecture["output_dim"], "Architecture output_dim"
            )

        model_spec = validate_mapping(architecture.get("model"), "Architecture model")
        model, actual_output_dim = self.builder.build(model_spec, input_dim)
        if not isinstance(model, Structure):
            raise TypeError("Architecture model must build a Structure component")
        if expected_output is not None and actual_output_dim != expected_output:
            raise ValueError(
                f"Architecture declares output_dim={expected_output}, "
                f"but builder inferred {actual_output_dim} features"
            )

        return Network(name, model, input_dim, actual_output_dim)
