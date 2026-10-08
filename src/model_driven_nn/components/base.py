from abc import ABC, abstractmethod

from shared.types import FloatArray

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
