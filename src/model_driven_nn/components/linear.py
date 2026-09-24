import numpy as np

from src.types import FloatArray

from ..parameter import Parameter
from .base import Layer


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.W = Parameter.from_data(np.random.randn(input_dim, output_dim))
        self.b = Parameter.from_data(np.zeros(output_dim))

        self.Y: FloatArray | None = None
        self.x: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        self.x = x
        self.Y = np.dot(x, self.W.data) + self.b.data
        return self.Y

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self.x is None:
            raise RuntimeError("Linear.backward() called before forward()")
        self.W.grad = self.x.T @ d_y
        self.b.grad = np.sum(d_y, axis=0)
        return d_y @ self.W.data.T

    def params(self) -> list[Parameter]:
        return [self.W, self.b]
