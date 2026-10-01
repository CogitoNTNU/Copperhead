import numpy as np


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, params):
        for p in params:
            p.data -= self.learning_rate * p.grad


class AdaGrad:
    def __init__(self, learning_rate, eps):
        self.learning_rate = learning_rate
        self.eps = eps
        self.G = {}

    def step(self, params):
        for i in range(len(params)):
            if i not in self.G:
                self.G[i] = np.zeros_like(params[i].data)

            p = params[i]
            G_i = self.G[i]
            G_i += p.grad**2

            p.data -= self.learning_rate * p.grad / (np.sqrt(G_i) + self.eps)
