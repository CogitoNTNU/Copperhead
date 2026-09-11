from .components import Sequential
from .types import FloatArray, ParameterPair


class Network:
    def __init__(
        self, name: str, model: Sequential, input_dim: int, output_dim: int
    ) -> None:
        self.name: str = name
        self.model: Sequential = model
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

    def forward(self, x: FloatArray) -> FloatArray:
        return self.model.forward(x)

    def backward(self, d_y: FloatArray) -> FloatArray:
        return self.model.backward(d_y)

    def params(self) -> list[ParameterPair]:
        return self.model.params()
