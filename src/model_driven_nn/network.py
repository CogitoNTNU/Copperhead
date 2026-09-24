from .components import Structure
from .parameter import Parameter
from .types import FloatArray


class Network:
    def __init__(
        self, name: str, model: Structure, input_dim: int, output_dim: int
    ) -> None:
        self.name: str = name
        self.model: Structure = model
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

    def forward(self, x: FloatArray) -> FloatArray:
        return self.model.forward(x)

    def backward(self, d_y: FloatArray) -> FloatArray:
        return self.model.backward(d_y)

    def params(self) -> list[Parameter]:
        return self.model.params()
