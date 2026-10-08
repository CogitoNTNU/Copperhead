from personal.collab.mnist_nn import SGD, Linear, ReLU, Sequential, rng
from src.training.mnist_train import train


def main():
    network = Sequential(Linear(784, 64, rng), ReLU(), Linear(64, 10, rng))
    optimizer = SGD(learning_rate=0.05)

    train(
        nn=network,
        optimizer=optimizer,
        engine_name="collab",
    )


if __name__ == "__main__":
    main()
