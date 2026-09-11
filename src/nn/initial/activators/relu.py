import numpy as np

from src.nn.initial.activators.activator import Activator


class ReLU(Activator):
    def __init__(self):
        super().__init__("ReLU")
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        return d_y * self.mask
