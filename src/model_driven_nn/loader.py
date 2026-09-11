from pathlib import Path

import yaml

from .builder import Builder, BuildResult


class Loader:
    def __init__(self, builder: Builder | None = None) -> None:
        self.builder = builder if builder is not None else Builder()

    def load(self, path: str | Path) -> BuildResult:
        with Path(path).open(encoding="utf-8") as file:
            spec = yaml.safe_load(file)

        if not isinstance(spec, dict):
            raise TypeError("Architecture must be a YAML mapping")

        if not all(isinstance(key, str) for key in spec):
            raise TypeError("Architecture keys must be strings")

        if spec.get("type") != "sequential":
            raise ValueError("Root component must be sequential")

        input_dim = self._positive_int(spec.get("input_dim"), "Root input_dim")

        expected_output = None
        if "output_dim" in spec:
            expected_output = self._positive_int(spec["output_dim"], "Root output_dim")

        network, output_dim = self.builder.build(spec, input_dim)

        if expected_output is not None and output_dim != expected_output:
            raise ValueError(
                f"Architecture declares output_dim={expected_output}, "
                f"but produces {output_dim} features"
            )

        return network, output_dim

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

        return value
