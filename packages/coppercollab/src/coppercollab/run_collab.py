from coppercore.training.mnist_train import train

from coppercollab.mnist_nn import SGD, Linear, ReLU, Sequential, rng


def main():
    network = Sequential(Linear(784, 64, rng), ReLU(), Linear(64, 10, rng))
    optimizer = SGD(learning_rate=0.05)

    train(
        network=network,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    main()
