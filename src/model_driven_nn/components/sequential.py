from collections.abc import Iterable

from ..types import FloatArray, ParameterPair
from .base import Component


class Sequential(Component):
    def __init__(self, components: Iterable[Component]) -> None:
        self.components: list[Component] = list(components)

    def forward(self, x: FloatArray) -> FloatArray:
        for component in self.components:
            x = component.forward(x)
        return x

    def backward(self, d_y: FloatArray) -> FloatArray:
        for component in reversed(self.components):
            d_y = component.backward(d_y)
        return d_y

    def params(self) -> list[ParameterPair]:
        return [
            parameter
            for component in self.components
            for parameter in component.params()
        ]
