import numpy as np

import wandb
from model_driven_nn import Network
from src.data.mnist_loader import load_mnist


def softmax_cross_entropy_loss(logits, y_true):
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


def accuracy(logits, y_true):
    preds = logits.argmax(axis=1)
    return (preds == y_true).mean()


def iterate_batches(X, y, batch_size, rng):
    n = len(X)
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        yield X[idx], y[idx]


def train(
    network: Network,
    optimizer,
    epochs=5,
    batch_size=64,
    seed=0,
):
    wandb.init(
        project="copperhead-mnist",
        name=network.name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": optimizer.learning_rate,
        },
    )

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
            optimizer.step(network.params())

            total_loss += loss * len(x_batch)
            total_examples += len(x_batch)

        val_logits = network.forward(X_val)
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
            f"epoch {epoch + 1} av {epochs}"
            f"training loss = {train_loss:.4f}"
            f"accuracy = {val_acc:.4f}"
        )

    wandb.finish()
    return network
