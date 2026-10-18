import numpy as np
import matplotlib.pyplot as plt
from src.data.mnist_loader import load_mnist

from src.oskarNN.nn.network import Network
from src.oskarNN.nn.optimizer import OskarSgd
from src.training.train_oskar.mnist_train_oskar import train
from src.oskarNN.nn.layers import Linear, ReLu
from src.oskarNN.nn.losses import softmax_cross_entropy


def show_prediction(network, X, y, index):
    image = X[index]
    label = y[index]

    logits = network.forward(image.reshape(1, -1))
    prediction = np.argmax(logits, axis=1)[0]

    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"Predikert: {prediction}, faktisk: {label}")
    plt.show()


def main():
    layers = [
        Linear(input_dim=784, output_dim=128, seed=0),
        ReLu(),
        Linear(input_dim=128, output_dim=10, seed=1),
    ]
    network = Network(layers)
    optimizer = OskarSgd(learning_rate=0.1)

    network = train(
        nn=network,
        optimizer=optimizer,
        loss_fn=softmax_cross_entropy,
        engine_name="oskar",
    )

    _, _, X_val, y_val = load_mnist()
    show_prediction(network, X_val, y_val, index=0)


if __name__ == "__main__":
    main()
