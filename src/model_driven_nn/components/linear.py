import numpy as np

from shared.types import FloatArray

from ..parameter import Parameter
from .base import Layer


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        scale = np.sqrt(2.0 / input_dim)

        self.W = Parameter.from_data(np.random.randn(input_dim, output_dim) * scale)
        self.b = Parameter.from_data(np.zeros(output_dim))

        self.x: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        self.x = x
        return np.dot(x, self.W.data) + self.b.data

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self.x is None:
            raise RuntimeError("Linear.backward() called before forward()")
        self.W.grad += self.x.T @ d_y
        self.b.grad += np.sum(d_y, axis=0)
        return d_y @ self.W.data.T

    def params(self) -> list[Parameter]:
        return [self.W, self.b]
