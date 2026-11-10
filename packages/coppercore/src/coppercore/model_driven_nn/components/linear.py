import numpy as np

from coppercore.shared import FloatArray

from ..parameter import Parameter
from .base import Layer


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        scale = np.sqrt(2.0 / input_dim)

        self.W = Parameter.from_data(np.random.randn(input_dim, output_dim) * scale)
        self.b = Parameter.from_data(np.zeros(output_dim))

        self.x: FloatArray | None = None

    def forward(
        self,
        x: FloatArray,
        *,
        cache: bool = True,
    ) -> FloatArray:
        if cache:
            self.x = x
        x_2d = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
        return np.dot(x_2d, self.W.data) + self.b.data

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self.x is None:
            raise RuntimeError("Linear.backward() called before forward()")
        x_2d = self.x.reshape(self.x.shape[0], -1) if self.x.ndim > 2 else self.x
        self.W.grad += x_2d.T @ d_y
        self.b.grad += np.sum(d_y, axis=0)
        d_x = d_y @ self.W.data.T
        if self.x.ndim > 2:
            return d_x.reshape(self.x.shape)
        return d_x

    def params(self) -> list[Parameter]:
        return [self.W, self.b]
