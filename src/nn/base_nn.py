import numpy as np


class your_nn:
    def __init__(self, in_features, out_features):
        self.in_features = 784
        self.out_features = 10
        self.W = np.zeros((in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def backward(self, grad_output):
        return grad_output @ self.W.T

    def params(self):
        return [(self.W, None), (self.b, None)]
