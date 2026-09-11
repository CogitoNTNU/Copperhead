from abc import ABC, abstractmethod


class Component(ABC):
    @abstractmethod
    def forward(self, x):
        """Compute the component output for x."""

    @abstractmethod
    def backward(self, d_y):
        """Return the gradient with respect to the component input."""

    def params(self):
        return []


class Layer(Component, ABC):
    """Marker base class for layers."""


class Activation(Component, ABC):
    """Marker base class for activations."""
