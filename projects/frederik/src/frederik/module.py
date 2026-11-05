import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self) -> None:
        return self.grad.fill(0)


class Module:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def params(self):
        return []
