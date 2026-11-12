from abc import ABC, abstractmethod
from typing import Any, Self

from coppercore.shared import FloatArray, YamlMapping

from ..parameter import Parameter


class Component(ABC):
    @abstractmethod
    def forward(
        self,
        x: FloatArray,
        *,
        cache: bool = True,
    ) -> FloatArray:
        """Compute the component output for x."""

    @abstractmethod
    def backward(self, d_y: FloatArray) -> FloatArray:
        """Return the gradient with respect to the component input."""

    def params(self) -> list[Parameter]:
        return []

    def to_config(self) -> YamlMapping:
        raise NotImplementedError(
            f"{type(self).__name__} does not support serialization to config"
        )

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        return []

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        for param in self.params():
            param.to(device=device, dtype=dtype)
        return self


class Layer(Component, ABC):
    """Marker base class for layers."""


class Activation(Component, ABC):
    """Marker base class for activations."""


class Structure(Component, ABC):
    """Marker base class for model structures."""

    @abstractmethod
    def children(self) -> list[Component]: ...

    def params(self) -> list[Parameter]:
        return [parameter for child in self.children() for parameter in child.params()]

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        for child in self.children():
            child.to(device=device, dtype=dtype)
        return self
