from optimizer.optimizer import Optimizer


class SGD(Optimizer):
    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad
