import numpy as np

from src.nn.initial.activators import Act
from src.nn.initial.activators.activator import Activator
from src.nn.initial.linear import Linear

type Layer = Linear | Activator


class InitNN:
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 2048,
        output_dim: int = 10,
    ) -> None:
        self.layers: list[Layer] = [
            Linear(input_dim, hidden_dim),
            Act.ReLU(),
            Linear(hidden_dim, hidden_dim),
            Act.ReLU(),
            Linear(hidden_dim, output_dim),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            d_y = layer.backward(d_y)
        return d_y

    def params(self):
        pass
