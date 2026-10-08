from abc import ABC, abstractmethod

import numpy as np


class Activator(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, d_y: np.ndarray) -> np.ndarray:
        pass
