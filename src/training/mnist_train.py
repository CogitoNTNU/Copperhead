from collections.abc import Iterator

import numpy as np

import wandb
from data.mnist_loader import load_mnist
from model_driven_nn import Network
from optimizer.optimizer import Optimizer
from shared.types import FloatArray, IntArray


def softmax_cross_entropy_loss(
    logits: FloatArray,
    y_true: IntArray,
) -> tuple[float, FloatArray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    n = logits.shape[0]

    log_likelihood = -np.log(probs[np.arange(n), y_true] + 1e-12)

    loss = log_likelihood.mean()

    grad = probs.copy()
    grad[np.arange(n), y_true] -= 1
    grad /= n
    return loss, grad


def accuracy(
    logits: FloatArray,
    y_true: IntArray,
) -> float:
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
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 0,
) -> Network:
    wandb.init(
        project="copperhead-mnist",
        name=network.name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": optimizer.learning_rate,
        },
    )

    optimizer.parameters = network.params()

    X_train, y_train, X_val, y_val = load_mnist(seed=seed)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0
        for x_batch, y_batch in iterate_batches(X_train, y_train, batch_size, rng):
            network.zero_grad()

            logits = network.forward(x_batch)
            loss, grad_loss = softmax_cross_entropy_loss(logits, y_batch)

            network.backward(grad_loss)
            optimizer.step()

            total_loss += loss * len(x_batch)
            total_examples += len(x_batch)

        val_logits = network.predict(X_val)
        val_acc = accuracy(val_logits, y_val)
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
