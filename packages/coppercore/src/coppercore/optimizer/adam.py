from coppercore.optimizer.optimizer import Optimizer
import numpy as np


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

        self.moment_1 = []  # list of arrays
        self.moment_2 = []

    def step(self) -> None:
        self.time_step += 1
        # first time step is called, create zero arrays of shape like parameters
        if not self.moment_1:  # empty list = false
            self.moment_1 = [np.zeros_like(p.data) for p in self.parameters]
            self.moment_2 = [np.zeros_like(p.data) for p in self.parameters]

        for i, parameter in enumerate(self.parameters):
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
                / (np.sqrt(moment_2_estimate) + self.eps)
            )

            parameter.data += update
