import numpy as np
from coppercore.training.mnist_train import train

from frederik.activation import ReLU
from frederik.linear import Linear
from frederik.model import Sequential
from frederik.optimizer import AdaGrad


def main():
    rng = np.random.default_rng(seed=42)
    network = Sequential(
        Linear(784, 64, rng), ReLU(), Linear(64, 64, rng), ReLU(), Linear(64, 10, rng)
    )
    optimizer = AdaGrad(learning_rate=0.01, eps=1e-8)

    train(nn=network, optimizer=optimizer, engine_name="frederik", epochs=50)


if __name__ == "__main__":
    main()
