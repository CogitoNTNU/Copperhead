import numpy as np


class Linear:
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
        self.grad_W = np.dot(self.x.T, d_y)  # ty: ignore[unresolved-attribute]
        self.grad_b = np.sum(d_y, axis=0)
        return np.dot(d_y, self.W.T)
