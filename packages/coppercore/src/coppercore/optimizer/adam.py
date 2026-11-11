from typing import Any, Self

import numpy as np

from coppercore.backend import get_array_module, to_device
from coppercore.optimizer.optimizer import Optimizer


class ADAM(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        moment_1_decay: float = 0.9,
        moment_2_decay: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)

        self.moment_1_decay = moment_1_decay
        self.moment_2_decay = moment_2_decay
        self.eps = eps
        self.time_step = 0

        self.moment_1: list[Any] = []  # list of arrays
        self.moment_2: list[Any] = []

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        """Migrate moment vectors to target device and/or dtype."""
        if self.moment_1:
            self.moment_1 = [
                to_device(m, device=device, dtype=dtype) for m in self.moment_1
            ]
        if self.moment_2:
            self.moment_2 = [
                to_device(m, device=device, dtype=dtype) for m in self.moment_2
            ]
        return self

    def step(self) -> None:
        self.time_step += 1
        # first time step is called, create zero arrays of shape like parameters
        if not self.moment_1 or len(self.moment_1) != len(self.parameters):
            self.moment_1 = [
                get_array_module(p.data).zeros_like(p.data) for p in self.parameters
            ]
            self.moment_2 = [
                get_array_module(p.data).zeros_like(p.data) for p in self.parameters
            ]

        for i, parameter in enumerate(self.parameters):
            xp = get_array_module(parameter.data, parameter.grad)
            if get_array_module(self.moment_1[i]) is not xp:
                target_device = "cpu" if xp is np else "gpu"
                self.moment_1[i] = to_device(
                    self.moment_1[i],
                    device=target_device,
                    dtype=parameter.data.dtype,
                )
                self.moment_2[i] = to_device(
                    self.moment_2[i],
                    device=target_device,
                    dtype=parameter.data.dtype,
                )
            elif self.moment_1[i].dtype != parameter.data.dtype:
                self.moment_1[i] = self.moment_1[i].astype(parameter.data.dtype)
                self.moment_2[i] = self.moment_2[i].astype(parameter.data.dtype)

            grad = parameter.grad
            # update bias first moment estimate
            self.moment_1[i] *= self.moment_1_decay
            self.moment_1[i] += (1 - self.moment_1_decay) * grad

            self.moment_2[i] *= self.moment_2_decay
            self.moment_2[i] += ((1 - self.moment_2_decay) * grad) * grad
            # haramard product here

            # correct biases in moment 1 and 2
            moment_1_estimate = self.moment_1[i] / (
                1 - self.moment_1_decay**self.time_step
            )
            moment_2_estimate = self.moment_2[i] / (
                1 - self.moment_2_decay**self.time_step
            )

            update = (
                -self.learning_rate
                * moment_1_estimate
                / (xp.sqrt(moment_2_estimate) + self.eps)
            )

            parameter.data += update
