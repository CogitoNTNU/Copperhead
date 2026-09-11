from abc import ABC, abstractmethod

from ..types import FloatArray, ParameterPair


class Component(ABC):
    @abstractmethod
    def forward(self, x: FloatArray) -> FloatArray:
        """Compute the component output for x."""

    @abstractmethod
    def backward(self, d_y: FloatArray) -> FloatArray:
        """Return the gradient with respect to the component input."""

    def params(self) -> list[ParameterPair]:
        return []


class Layer(Component, ABC):
    """Marker base class for layers."""


class Activation(Component, ABC):
    """Marker base class for activations."""
