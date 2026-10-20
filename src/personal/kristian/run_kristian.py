from src.nn.kristian.nn import Network
from src.nn.kristian.optimizer import KristianSgd

from src.training.mnist_train import train


def main():
    network = Network()
    optimizer = KristianSgd(learning_rate=0.1)

    train(nn=network, optimizer=optimizer, engine_name="kristian", epochs=15)


if __name__ == "__main__":
    main()
