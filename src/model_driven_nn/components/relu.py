import numpy as np

from .base import Activation


class ReLU(Activation):
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward() called before forward()")
        return d_y * self.mask
