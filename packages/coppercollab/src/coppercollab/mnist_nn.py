import numpy as np


class Parameter:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = np.zeros_like(data)


class Module:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def params(self):
        return []


def default_init(rng: np.random.Generator, shape: tuple, lim) -> np.ndarray:
    return rng.uniform(-lim, lim, shape)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        w_shape = (out_features, in_features)
        b_shape = (out_features,)

        self.weight = Parameter(default_init(rng, w_shape, 1))
        self.bias = Parameter(default_init(rng, b_shape, 1))

        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        pred = (
            self.x @ self.weight.data.T + self.bias.data
        )  # (, in) x (in, out) = (, out)
        return pred

    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        """
        dL/dW = dL/da * da/dz * dz/dW
        a = relu(z)
        z = wx+b
        """

        self.weight.grad = gradient_output.T @ self.x  # (out, B) x (B, in) = (out, in)
        self.bias.grad = np.sum(gradient_output, axis=0)  # (B, out)

        return gradient_output @ self.weight.data  # (B, out) x (out, in) = (B, in)

    def params(self):
        return [p for p in (self.weight, self.bias)]


class ReLU(Module):
    def __init__(self) -> None:
        self.x = None
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        out = np.maximum(self.x, 0)
        self.mask = out > 0

        return out

    def backward(self, grad_output) -> np.ndarray:
        return grad_output * self.mask

    def params(self):
        return []


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self.modules = list(modules)

    def forward(self, x):
        out = x

        for m in self.modules:
            out = m.forward(out)

        return out

    def backward(self, gradient_output):
        out = gradient_output

        for m in reversed(self.modules):
            out = m.backward(out)

        return out

    def params(self):
        return [p for m in self.modules for p in m.params()]


class SGD:
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def step(self, params):
        for p in params:
            p.data -= self.learning_rate * p.grad


rng = np.random.default_rng(seed=42)
network = Sequential(Linear(784, 64, rng), ReLU(), Linear(64, 10, rng))


# training loop

# valuation

# data load

# main script
