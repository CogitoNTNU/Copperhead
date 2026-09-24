import numpy as np
from src.nn.frederik.module import Module


class ReLU(Module):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """x: (batch, n) -> (batch, n). Save mask for backward."""

        sigma = np.maximum(0, x)
        self.mask = sigma > 0  # boolean array

        return sigma

    def backward(self, dout):
        return dout * self.mask

    def parameters(self):
        return []
