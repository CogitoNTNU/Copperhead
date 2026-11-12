from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coppercore.optimizer.optimizer import Optimizer

if TYPE_CHECKING:
    from coppercore.model_driven_nn.network import Network


class SGD(Optimizer):
    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad

    def state_dict(self, network: Network) -> dict[str, Any]:
        return {
            "optimizer_type": "SGD",
            "learning_rate": self.learning_rate,
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        network: Network,
    ) -> None:
        if state_dict["optimizer_type"] != "SGD":
            raise ValueError("Checkpoint is not an SGD optimizer")

        self.learning_rate = float(state_dict["learning_rate"])
