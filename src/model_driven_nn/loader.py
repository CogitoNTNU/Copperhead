from collections.abc import Mapping
from pathlib import Path

from .builder import Builder
from .configuration import ConfigurationLoader
from .common import positive_int, validate_mapping
from .components import Structure
from .network import Network
from .types import ArchitectureSpec


class Loader:
    def __init__(self, builder: Builder | None = None) -> None:
        self.builder = builder if builder is not None else Builder()
        self.configuration = ConfigurationLoader()

    def load(
        self, path: str | Path, config: Mapping[str, object] | None = None
    ) -> Network:
        architecture = self.load_spec(path, config=config)
        name = architecture.get("name")
        input_dim = architecture["input_dim"]
        expected_output = architecture.get("output_dim")
        model_spec = architecture["model"]
        model, actual_output_dim = self.builder.build(model_spec, input_dim)
        if not isinstance(model, Structure):
            raise TypeError("Architecture model must build a Structure component")
        if expected_output is not None and actual_output_dim != expected_output:
            raise ValueError(
                f"Architecture declares output_dim={expected_output}, "
                f"but builder inferred {actual_output_dim} features"
            )

        return Network(name, model, input_dim, actual_output_dim)

    def load_spec(
        self, path: str | Path, config: Mapping[str, object] | None = None
    ) -> ArchitectureSpec:
        """Load and resolve an architecture without constructing its network."""
        spec = self.configuration.load(path, config=config)
        architecture = validate_mapping(spec, "Architecture")

        name = architecture.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Architecture name must be a nonblank string")
        positive_int(architecture.get("input_dim"), "Architecture input_dim")
        if "output_dim" in architecture:
            positive_int(architecture["output_dim"], "Architecture output_dim")
        validate_mapping(architecture.get("model"), "Architecture model")
        return architecture
