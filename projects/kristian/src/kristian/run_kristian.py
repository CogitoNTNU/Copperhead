from src.training.mnist_train import train

from kristian.network import KristianNN
from kristian.optimizer import KristianSgd


def main():
    network = Network()
    optimizer = KristianSgd(learning_rate=0.1)

    train(nn=network, optimizer=optimizer, engine_name="kristian", epochs=15)


if __name__ == "__main__":
    main()
