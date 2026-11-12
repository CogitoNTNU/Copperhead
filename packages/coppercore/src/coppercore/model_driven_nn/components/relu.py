from typing import Any, Self

from coppercore.backend import get_array_module, to_device
from coppercore.shared import BoolArray, FloatArray, YamlMapping

from .base import Activation


class ReLU(Activation):
    def __init__(self) -> None:
        self.mask: BoolArray | None = None

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        super().to(device=device, dtype=dtype)
        if self.mask is not None:
            self.mask = to_device(self.mask, device=device)
        return self

    def forward(
        self,
        x: FloatArray,
        *,
        cache: bool = True,
    ) -> FloatArray:
        if cache:
            self.mask = x > 0
        xp = get_array_module(x)
        return xp.maximum(x, 0)

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward() called before forward()")
        return d_y * self.mask

    def to_config(self) -> YamlMapping:
        return {"type": "relu"}
