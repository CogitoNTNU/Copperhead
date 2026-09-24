from collections.abc import Iterable

from src.types import FloatArray

from .base import Component, Structure


class Sequential(Structure):
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

    def children(self) -> list[Component]:
        return self.components
