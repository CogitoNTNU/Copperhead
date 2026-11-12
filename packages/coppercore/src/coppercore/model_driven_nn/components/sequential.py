from collections.abc import Iterable

from coppercore.shared import FloatArray, YamlMapping

from ..parameter import Parameter
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

    def to_config(self) -> YamlMapping:
        return {
            "type": "sequential",
            "layers": [component.to_config() for component in self.components],
        }

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        named_params = []
        for index, component in enumerate(self.components):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            named_params.extend(component.named_parameters(prefix=child_prefix))
        return named_params
