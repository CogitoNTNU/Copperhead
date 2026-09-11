import numpy as np

from ..types import FloatArray, ParameterPair
from .base import Layer


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.W: FloatArray = np.random.randn(input_dim, output_dim)
        self.b: FloatArray = np.zeros(output_dim)

        self.grad_W: FloatArray = np.zeros_like(self.W)
        self.grad_b: FloatArray = np.zeros_like(self.b)

        self.Y: FloatArray | None = None
        self.x: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        self.x = x
        self.Y = np.dot(x, self.W) + self.b
        return self.Y

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self.x is None:
            raise RuntimeError("Linear.backward() called before forward()")
        self.grad_W = self.x.T @ d_y
        self.grad_b = np.sum(d_y, axis=0)
        return d_y @ self.W.T

    def params(self) -> list[ParameterPair]:
        return [
            (self.W, self.grad_W),
            (self.b, self.grad_b),
        ]
