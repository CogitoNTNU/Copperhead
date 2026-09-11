import numpy as np

from ..types import BoolArray, FloatArray
from .base import Activation


class ReLU(Activation):
    def __init__(self) -> None:
        self.mask: BoolArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward() called before forward()")
        return d_y * self.mask
