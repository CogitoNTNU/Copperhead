from coppercore.training.mnist_train import train

from kristian.nn import Network
from kristian.optimizer import KristianSgd


def main():
    network = Network("kristian")
    optimizer = KristianSgd(learning_rate=0.1)

    train(network=network, optimizer=optimizer, epochs=15)


if __name__ == "__main__":
    main()
