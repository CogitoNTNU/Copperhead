from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from coppercore.shared import FloatArray, YamlMapping, format_number

from .components import Structure
from .parameter import Parameter


class Network:
    def __init__(
        self, name: str, model: Structure, input_dim: int, output_dim: int
    ) -> None:
        self.model: Structure = model
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        formatted = self.formatted_number()
        if formatted and not name.endswith(f"_{formatted}"):
            name = f"{name}_{formatted}"
        self.name: str = name

    def number(self) -> int:
        return sum(param.data.size for param in self.params())

    def formatted_number(self) -> str:
        return format_number(self.number())

    def predict(self, x: FloatArray) -> FloatArray:
        return self.forward(x, cache=False)

    def forward(self, x: FloatArray, *, cache: bool = True) -> FloatArray:
        return self.model.forward(x, cache=cache)

    def backward(self, d_y: FloatArray) -> FloatArray:
        return self.model.backward(d_y)

    def params(self) -> list[Parameter]:
        return self.model.params()

    def zero_grad(self) -> None:
        for param in self.params():
            param.zero_grad()

    def named_parameters(self, prefix: str = "model") -> list[tuple[str, Parameter]]:
        return self.model.named_parameters(prefix=prefix)

    def state_dict(self) -> dict[str, FloatArray]:
        from .serialization import extract_state_dict

        return extract_state_dict(self.named_parameters())

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        from .serialization import validate_and_load_state_dict

        validate_and_load_state_dict(self.named_parameters(), state_dict)

    def save_weights(self, path: str | Path) -> None:
        from .serialization import save_weights_to_file

        save_weights_to_file(path, self.state_dict())

    def load_weights(self, path: str | Path) -> None:
        from .serialization import load_weights_from_file

        loaded = load_weights_from_file(path)
        self.load_state_dict(loaded)

    def to_config(self) -> YamlMapping:
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "model": self.model.to_config(),
        }

    def save_architecture(self, path: str | Path) -> None:
        from .serialization import save_architecture_to_file

        save_architecture_to_file(path, self.to_config())

    def save_model(self, path: str | Path) -> None:
        from .serialization import save_model_bundle

        save_model_bundle(self, path)

    @classmethod
    def load_model(
        cls,
        path: str | Path,
        device: str | None = None,
        dtype: Any = None,
    ) -> "Network":
        from .serialization import load_model_bundle

        return load_model_bundle(path, device=device, dtype=dtype)

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        self.model.to(device=device, dtype=dtype)
        return self
