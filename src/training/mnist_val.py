import numpy as np

from src.data.mnist_loader import load_mnist
from src.nn.kristian import skeleton_nn
from src.training.mnist_train import accuracy


def validate(weights_path="data/basic_nn_weights.npz", seed=0):
    _, _, X_val, y_val = load_mnist(seed=seed)

    nn = skeleton_nn()
    saved = np.load(weights_path)
    nn.W[:] = saved["W"]
    nn.b[:] = saved["b"]

    logits = nn.forward(X_val)
    acc = accuracy(logits, y_val)
    print(f"val accuracy {acc:.4f}")
    return acc


if __name__ == "__main__":
    validate()
