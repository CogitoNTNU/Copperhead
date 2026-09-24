import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


class Module:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def parameters(self):
        return []
