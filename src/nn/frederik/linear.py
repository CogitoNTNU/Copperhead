import numpy as np
from src.nn.frederik.module import Parameter, Module
from src.nn.frederik.init import default_init


class Linear(Module):
    """Performs a linear transformation"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        use_bias: bool = True,
        dtype=np.float64,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `rng`: A NumPy random number generator"""

        w_shape = (out_features, in_features)
        b_shape = (out_features,)

        # interval for random generation
        if in_features == 0:
            lim = 1.0
        else:
            lim = 1 / np.sqrt(in_features)

        self.weight = Parameter(
            default_init(rng, w_shape, lim, dtype)
        )  # shape: (out, in)
        self.bias = (
            Parameter(default_init(rng, b_shape, lim, dtype)) if use_bias else None
        )  # shape: (B, out)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.x = None  # shape: (B, in)

    def forward(self, x):
        """x: (batch, in) -> (batch, out). Save for backward."""
        assert x.ndim == 2 and x.shape[1] == self.in_features

        self.x = x
        out = x @ self.weight.data.T

        if self.bias is not None:
            out += self.bias.data

        return out

    def backward(self, dout):
        """dout: (batch, out) -> dx: (batch, in)"""
        # here dout = delta^L
        self.weight.grad = dout.T @ self.x  # sum(sigma(z_k^{l-1}) * delta_j^l)

        if self.bias is not None:
            self.bias.grad = dout.sum(axis=0)  # sum(delta_j^l)

        return dout @ self.weight.data  # partial C / partial z^{L-1}

    def params(self):
        return [p for p in (self.weight, self.bias) if p is not None]
