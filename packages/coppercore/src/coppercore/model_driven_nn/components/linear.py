from typing import Any, Self

import numpy as np

from coppercore.backend import get_array_module, to_device
from coppercore.shared import FloatArray, YamlMapping

from ..parameter import Parameter
from .base import Layer


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        scale = np.sqrt(2.0 / input_dim)

        self.W = Parameter.from_data(np.random.randn(input_dim, output_dim) * scale)
        self.b = Parameter.from_data(np.zeros(output_dim))

        self.x: FloatArray | None = None

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        super().to(device=device, dtype=dtype)
        if self.x is not None:
            self.x = to_device(self.x, device=device, dtype=dtype)
        return self

    def forward(
        self,
        x: FloatArray,
        *,
        cache: bool = True,
    ) -> FloatArray:
        if cache:
            self.x = x
        x_2d = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
        xp = get_array_module(x_2d, self.W.data)
        return xp.dot(x_2d, self.W.data) + self.b.data

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self.x is None:
            raise RuntimeError("Linear.backward() called before forward()")
        xp = get_array_module(d_y, self.W.data)
        x_2d = self.x.reshape(self.x.shape[0], -1) if self.x.ndim > 2 else self.x
        self.W.grad += xp.dot(x_2d.T, d_y)
        self.b.grad += xp.sum(d_y, axis=0)
        d_x = xp.dot(d_y, self.W.data.T)
        if self.x.ndim > 2:
            return d_x.reshape(self.x.shape)
        return d_x

    def params(self) -> list[Parameter]:
        return [self.W, self.b]

    def to_config(self) -> YamlMapping:
        return {
            "type": "linear",
            "output_dim": int(self.W.data.shape[1]),
        }

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        p_w = f"{prefix}.W" if prefix else "W"
        p_b = f"{prefix}.b" if prefix else "b"
        return [(p_w, self.W), (p_b, self.b)]
