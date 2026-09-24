import numpy as np

from src.oskarNN.nn.layers import Linear, ReLu
from src.oskarNN.nn.losses import softmax_cross_entropy


def check_loss():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 10))
    y_true = rng.integers(0, 10, size=5)

    loss, grad = softmax_cross_entropy(logits, y_true)

    def loss_fn(logits_):
        loss_, _ = softmax_cross_entropy(logits_, y_true)
        return loss_

    numeric_grad = numerical_gradient(loss_fn, logits.copy())

    print("Loss grad max error:", relative_error(grad, numeric_grad).max())


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + eps
        f_plus = f(x)

        x[idx] = original - eps
        f_minus = f(x)

        x[idx] = original

        grad[idx] = (f_plus - f_minus) / (2 * eps)

    return grad


def relative_error(analytic, numeric):
    return np.abs(analytic - numeric) / (np.abs(analytic) + np.abs(numeric) + 1e-8)


def check_linear():
    rng = np.random.default_rng(0)
    layer = Linear(input_dim=4, output_dim=3, seed=0)

    x = rng.standard_normal((5, 4))
    grad_output = rng.standard_normal((5, 3))

    layer.forward(x)
    grad_x = layer.backward((grad_output))

    def loss_x(x_):
        return np.sum(layer.forward(x_) * grad_output)

    def loss_W(W_):
        layer.W = W_
        return np.sum(layer.forward(x) * grad_output)

    def loss_b(b_):
        layer.b = b_
        return np.sum(layer.forward(x) * grad_output)

    numeric_grad_x = numerical_gradient(loss_x, x.copy())
    numeric_grad_W = numerical_gradient(loss_W, layer.W.copy())
    numeric_grad_b = numerical_gradient(loss_b, layer.b.copy())

    print("Linear grad_x max error:", relative_error(grad_x, numeric_grad_x).max())
    print(
        "Linear grad_W max error:", relative_error(layer.grad_W, numeric_grad_W).max()
    )
    print(
        "Linear grad_b max error:", relative_error(layer.grad_b, numeric_grad_b).max()
    )


def check_relu():
    rng = np.random.default_rng(0)
    layer = ReLu()

    x = rng.standard_normal((5, 4))
    grad_output = rng.standard_normal((5, 4))

    layer.forward(x)
    grad_x = layer.backward(grad_output)

    def loss_x(x_):
        return np.sum(layer.forward(x_) * grad_output)

    numeric_grad_x = numerical_gradient(loss_x, x.copy())

    print("ReLu grad_x max error:", relative_error(grad_x, numeric_grad_x).max())


if __name__ == "__main__":
    check_linear()
    check_relu()
    check_loss()
