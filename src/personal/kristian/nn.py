import numpy as np


class KristianLinear:
    # seed can be whatever just needs to be standardized
    def __init__(self, in_features: int, out_features: int, seed=0) -> None:
        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((in_features, out_features)) * 0.01
        self.b = np.zeros(out_features)

        self.x = None
        self.Y = None

        self.grad_W = None
        self.grad_b = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return (self.x @ self.W) + self.b

    def backward(self, grad_output) -> None:
        self.grad_W = self.x.T @ grad_output

    def params(self) -> list:
        # class exists to return gradients weights and biases to optimizer and train
        return [(self.W, self.grad_W), (self.b, self.grad_b)]


class Network:
    def __init__(self):
        self.input_layer = KristianLinear(728, 4)
        self.hidden_layer1 = KristianLinear(4, 4)
        self.output_layer = KristianLinear(4, 10)

    def ReLU(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def forward(self, x):
        x = self.input_layer.forward(x)
        x = self.ReLU(x)
        x = self.hidden_layer1.forward(x)
        x = self.ReLU(x)
        x = self.output_layer.forward(x)
        return x

    def params(self) -> list:
        layers = (self.input_layer, self.hidden_layer1, self.output_layer)

        params = [pair for layer in layers for pair in layer.params()]

        return params
