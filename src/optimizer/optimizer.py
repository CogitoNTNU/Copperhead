from abc import ABC, abstractmethod
from collections.abc import Iterable

from model_driven_nn.parameter import Parameter


class Optimizer(ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self._parameters: list[Parameter] | None = None

    @property
    def parameters(self) -> list[Parameter]:
        if not self._parameters:
            raise RuntimeError("Optimizer parameters have not been set")
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Iterable[Parameter]) -> None:
        self._parameters = list(parameters)

    @abstractmethod
    def step(self) -> None:
        pass
