from src.nn.kristian.network import KristianNN
from src.nn.kristian.optimizer import KristianSgd
from src.training.mnist_train import train


def main():
    network = KristianNN(input_dim=784, output_dim=10, seed=0)
    optimizer = KristianSgd(learning_rate=0.1)

    train(
        nn=network,
        optimizer=optimizer,
        engine_name="kristian",
    )


if __name__ == "__main__":
    main()
