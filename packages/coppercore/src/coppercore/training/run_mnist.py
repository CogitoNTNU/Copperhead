import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from coppercore.backend import get_array_module, to_device
from coppercore.data import load_mnist
from coppercore.model_driven_nn import Network
from coppercore.model_driven_nn.construction import Loader
from coppercore.optimizer import SGD

from .mnist_train import accuracy, softmax_cross_entropy_loss, train

DEFAULT_NETWORK = Path(__file__).resolve().parents[1] / "networks" / "v1.yaml"


def evaluate(
    network: Network,
    *,
    seed: int,
    device: str | None = None,
    dtype: Any = None,
) -> tuple[float, float]:
    """Evaluate on the held-out split produced by ``load_mnist``."""
    _, _, x_val, y_val = load_mnist(seed=seed)

    params = network.params()
    if params:
        p_xp = get_array_module(params[0].data)
        param_device = "cpu" if p_xp is np else "gpu"
        param_dtype = params[0].data.dtype
    else:
        param_device = device or "cpu"
        param_dtype = dtype or np.float64

    x_val_dev = to_device(x_val, device=param_device, dtype=param_dtype)
    y_val_dev = to_device(y_val, device=param_device)

    logits = network.predict(x_val_dev)
    loss, _ = softmax_cross_entropy_loss(logits, y_val_dev)
    acc = accuracy(logits, y_val_dev)

    return float(loss), acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a YAML-defined network on MNIST."
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=DEFAULT_NETWORK,
        help="Path to the network YAML file.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Computation device (cpu or gpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Floating point precision (float32 or float64).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override the YAML hidden_dim value.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.hidden_dim is not None and args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    # Linear currently initializes through NumPy's global RNG.
    np.random.seed(args.seed)

    config = None
    if args.hidden_dim is not None:
        config = {"hidden_dim": args.hidden_dim}

    target_dtype = (
        np.float32
        if args.dtype == "float32" or (args.dtype is None and args.device == "gpu")
        else np.float64
    )

    network = Loader().load(args.network, config=config)
    network.to(device=args.device, dtype=target_dtype)

    optimizer = SGD(learning_rate=args.learning_rate)
    optimizer.to(device=args.device, dtype=target_dtype)

    print(
        f"training {network.name!r} | "
        f"epochs={args.epochs} | "
        f"batch_size={args.batch_size} | "
        f"lr={args.learning_rate:g} | "
        f"device={args.device} | "
        f"dtype={target_dtype.__name__}"
    )

    train(
        network,
        optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        dtype=target_dtype,
    )

    val_loss, val_acc = evaluate(
        network, seed=args.seed, device=args.device, dtype=target_dtype
    )
    print(f"final validation loss = {val_loss:.4f} | accuracy = {val_acc:.4f}")


if __name__ == "__main__":
    main()
