from coppercore.training.mnist_train import train

from kristian.nn import Network
from coppercore.optimizer.adam import ADAM


def main():
    network = Network("kristian_ADAM")
    optimizer = ADAM(learning_rate=0.001)

    train(network=network, optimizer=optimizer, epochs=15)


if __name__ == "__main__":
    main()
