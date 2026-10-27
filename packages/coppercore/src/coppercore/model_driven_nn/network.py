from coppercore.shared import FloatArray, format_number

from .components import Structure
from .parameter import Parameter


class Network:
    def __init__(
        self, name: str, model: Structure, input_dim: int, output_dim: int
    ) -> None:
        self.model: Structure = model
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        formatted = self.formatted_number()
        if formatted and not name.endswith(f"_{formatted}"):
            name = f"{name}_{formatted}"
        self.name: str = name

    def number(self) -> int:
        return sum(param.data.size for param in self.params())

    def formatted_number(self) -> str:
        return format_number(self.number())

    def predict(self, x: FloatArray) -> FloatArray:
        return self.forward(x, cache=False)

    def forward(self, x: FloatArray, *, cache: bool = True) -> FloatArray:
        return self.model.forward(x, cache=cache)

    def backward(self, d_y: FloatArray) -> FloatArray:
        return self.model.backward(d_y)

    def params(self) -> list[Parameter]:
        return self.model.params()

    def zero_grad(self) -> None:
        for param in self.params():
            param.zero_grad()
