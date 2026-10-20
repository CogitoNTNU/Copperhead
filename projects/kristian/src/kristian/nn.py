import numpy as np

seed = 0
rng = np.random.default_rng(seed)


class KristianLinear:
    # seed can be whatever just needs to be standardized
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        # use kaiming init to prevent dying relus
        # this init preserves variance across relu layers and has a mean of zero
        self.W = rng.normal(
            0, np.sqrt(2 / in_features), size=(in_features, out_features)
        )
        self.b = np.zeros(out_features)

        self.x = None
        self.Y = None

        self.grad_W = None
        self.grad_b = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        """
        x = (batch,in)
        w = (in,out)
        b = (batch,out)

        x * w = batch,out
        """
        return (self.x @ self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        dL/dW = dL/da * da/dz * dz/dW
        grad output is this layers output's gradient. dl/dlayer forward output = dl/dz
        so dz/dw = x, so self.x     z = wx+b
        and dl/dz = grad_output
        (in_features, out_features)= (batch_size, input_features) T * (batch_size, out_features)
        """
        self.grad_W = self.x.T @ grad_output

        """
        dL/db = dL/da * da/dz * dz/db
        dL/dz = grad output
        dz/db = 1
        multivariable chain rule says for f(x,y):
        df/ds = df/dx * dx/ds + df/dy * dy/ds
        so here we end up with a sum over all the grad outputs multiplied by one. Sum over all the rows. collapsing the batch dimention
        """

        self.grad_b = np.sum(grad_output, axis=0)  # (batch,out)

        """
        although what we previously computed and stored were dl/dw and dl/db this is not actially what we want to return.
        the point of them are actually for an optimizer to be able to acces them and use them in an optimizer step.

        what we want to send back is how much the previous layers output (their z, which is our x) affected the loss
        so therefore from this kth layer's point of view what we want to return in our dl/dx

        grad_output = dL/dz
        we need to return dL/dx = dL/dz * dz/dx, z =wx+b so 
        dl/dx = grad_output * w
        w = (in,out)
        grad = (batch,out)
        we wanbt dl/dx to have shape (batch,in), same as x
        so we do grad * w^T
        """
        return grad_output @ self.W.T

    def params(self) -> list:
        # class exists to return gradients weights and biases to optimizer and train
        return [(self.W, self.grad_W), (self.b, self.grad_b)]


class ReLU:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.maximum(0, x)
        self.mask = out > 0  # (Batch_size, features) boolean array
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask  # hadamard product

    def params(self) -> list:
        # class exists to return gradients weights and biases to optimizer and train
        # activation layer like ReLU has no learnable parameters so it should just return an empty list
        return []


class Network:
    def __init__(self, name: str):
        self.name = name
        self.input_layer = KristianLinear(784, 64)
        self.activation_layer1 = ReLU()
        self.hidden_layer1 = KristianLinear(64, 64)
        self.activation_layer = ReLU()
        self.output_layer = KristianLinear(64, 10)

        self.layers = [
            self.input_layer,
            self.activation_layer1,
            self.hidden_layer1,
            self.activation_layer,
            self.output_layer,
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, logit):
        for layer in reversed(self.layers):
            # we essentially need to do backward(backward(...)) so we just apply the function recursively and return it
            logit = layer.backward(logit)
        return logit

    def params(self) -> list:
        layers = (self.input_layer, self.hidden_layer1, self.output_layer)

        params = [pair for layer in layers for pair in layer.params()]
        """
        list comprehenstion version of 
        pair = []
        for layer in layers:
            for pair in layer:
                pair.append[layer.params()]
        """
        return params
