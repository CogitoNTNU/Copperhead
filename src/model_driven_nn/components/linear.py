import numpy as np

from .base import Layer


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.grad_b = None
        self.grad_W = None
        self.Y = None
        self.x = None
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.Y = np.dot(x, self.W) + self.b
        return self.Y

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("Linear.backward() called before forward()")
        self.grad_W = self.x.T @ d_y
        self.grad_b = np.sum(d_y, axis=0)
        return d_y @ self.W.T

    def params(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]
