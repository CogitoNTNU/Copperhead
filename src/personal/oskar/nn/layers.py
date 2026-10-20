import numpy as np


class Linear:
    def __init__(self, input_dim, output_dim, seed=0):
        rng = np.random.default_rng(seed)

        self.W = rng.standard_normal((input_dim, output_dim)) * 0.01
        self.b = np.zeros(output_dim)

        self.x = None

        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def params(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]


class ReLu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0)

    def params(self):
        return []
