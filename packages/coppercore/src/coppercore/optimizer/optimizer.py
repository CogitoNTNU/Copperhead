from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from coppercore.model_driven_nn.parameter import Parameter

if TYPE_CHECKING:
    from coppercore.model_driven_nn.network import Network


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

    def _resolve_named_parameters(
        self,
        network: Network,
    ) -> list[tuple[str, Parameter]]:
        named = list(network.named_parameters())
        optimizer_params = self.parameters

        names = [name for name, _ in named]
        network_ids = [id(param) for _, param in named]
        optimizer_ids = [id(param) for param in optimizer_params]

        if len(names) != len(set(names)):
            raise ValueError("Network contains duplicate parameter names")

        if len(network_ids) != len(set(network_ids)):
            raise ValueError("Network contains duplicate parameter objects")

        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError("Optimizer contains duplicate parameter objects")

        if set(network_ids) != set(optimizer_ids):
            raise ValueError("Optimizer parameters do not match network parameters")

        return named

    @abstractmethod
    def state_dict(self, network: Network) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        network: Network,
    ) -> None:
        pass

    def save_state(
        self,
        path: str | Path,
        network: Network,
    ) -> None:
        from .serialization import save_optimizer_state_to_file

        state = self.state_dict(network)
        save_optimizer_state_to_file(path, state)

    def load_state(
        self,
        path: str | Path,
        network: Network,
    ) -> None:
        from .serialization import load_optimizer_state_from_file

        state = load_optimizer_state_from_file(path)
        self.load_state_dict(state, network)

    @abstractmethod
    def step(self) -> None:
        pass

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        """Migrate internal optimizer state to target device and/or dtype."""
        return self
