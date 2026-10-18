from src.training.mnist_train import train
from src.nn.collab.mnist_nn import Sequential, Linear, rng, ReLU, SGD


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
