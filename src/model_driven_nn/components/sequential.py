from collections.abc import Iterable

from shared.types import FloatArray

from .base import Component, Structure


class Sequential(Structure):
    def __init__(self, components: Iterable[Component]) -> None:
        self.components: list[Component] = list(components)

    def forward(
        self,
        x: FloatArray,
        *,
        cache: bool = True,
    ) -> FloatArray:
        for component in self.components:
            x = component.forward(x, cache=cache)
        return x

    def backward(self, d_y: FloatArray) -> FloatArray:
        for component in reversed(self.components):
            d_y = component.backward(d_y)
        return d_y

    def children(self) -> list[Component]:
        return self.components
