from coppercore.optimizer.optimizer import Optimizer


class KristianSgd(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        super().__init__(learning_rate)

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad
