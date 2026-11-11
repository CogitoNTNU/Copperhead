from collections.abc import Iterator
from typing import Any

import numpy as np

import wandb
from coppercore.backend import get_array_module, to_device
from coppercore.data import load_mnist
from coppercore.model_driven_nn import Network
from coppercore.optimizer.optimizer import Optimizer
from coppercore.shared import FloatArray, IntArray


def softmax_cross_entropy_loss(
    logits: FloatArray,
    y_true: IntArray,
) -> tuple[Any, FloatArray]:
    xp = get_array_module(logits, y_true)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = xp.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    n = logits.shape[0]
    rows = xp.arange(n)

    log_likelihood = -xp.log(probs[rows, y_true] + 1e-12)

    loss = log_likelihood.mean()

    grad = probs.copy()
    grad[rows, y_true] -= 1
    grad /= n
    return loss, grad


def accuracy(
    logits: FloatArray,
    y_true: IntArray,
) -> float:
    get_array_module(logits, y_true)
    preds = logits.argmax(axis=1)
    return float((preds == y_true).mean())


def iterate_batches(
    X: FloatArray,
    y: IntArray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[FloatArray, IntArray]]:
    n = len(X)
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        yield X[idx], y[idx]


def train(
    network: Network,
    optimizer: Optimizer,
    epochs: int = 100,
    batch_size: int = 64,
    seed: int = 0xC0FFE,
    device: str | None = None,
    dtype: Any = None,
) -> Network:
    if device is not None or dtype is not None:
        network.to(device=device, dtype=dtype)
        optimizer.to(device=device, dtype=dtype)

    wandb.init(
        project="copperhead-mnist",
        name=network.name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": optimizer.learning_rate,
            "device": device or "cpu",
            "dtype": str(dtype) if dtype is not None else "float64",
        },
    )

    optimizer.parameters = network.params()

    X_train, y_train, X_val, y_val = load_mnist(seed=seed)
    rng = np.random.default_rng(seed)

    params = network.params()
    if params:
        p_xp = get_array_module(params[0].data)
        param_device = "cpu" if p_xp is np else "gpu"
        param_dtype = params[0].data.dtype
    else:
        param_device = device or "cpu"
        param_dtype = dtype or np.float64

    X_val_dev = to_device(X_val, device=param_device, dtype=param_dtype)
    y_val_dev = to_device(y_val, device=param_device)

    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0
        for x_batch, y_batch in iterate_batches(X_train, y_train, batch_size, rng):
            network.zero_grad()

            x_batch_dev = to_device(x_batch, device=param_device, dtype=param_dtype)
            y_batch_dev = to_device(y_batch, device=param_device)

            logits = network.forward(x_batch_dev)
            loss, grad_loss = softmax_cross_entropy_loss(logits, y_batch_dev)

            network.backward(grad_loss)
            optimizer.step()

            total_loss += loss * len(x_batch)
            total_examples += len(x_batch)

        val_logits = network.predict(X_val_dev)
        val_acc = accuracy(val_logits, y_val_dev)
        train_loss = total_loss / total_examples

        wandb.log(
            {
                "train_loss": train_loss,
                "learning_rate": optimizer.learning_rate,
                "val_acc": val_acc,
                "epoch": epoch,
            }
        )

        print(
            f"epoch {epoch + 1} av {epochs} | "
            f"training loss = {train_loss:.4f} | "
            f"accuracy = {val_acc:.4f}"
        )

    wandb.finish()
    return network
