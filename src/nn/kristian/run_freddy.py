import numpy as np

from src.nn.kristian.model import Sequential
from src.nn.kristian.linear import Linear
from src.nn.frederik.activation import ReLU
from src.nn.kristian.loss import MSE_loss
from src.nn.kristian.optimizer import SGD


def main():
    rng = np.random.default_rng(seed=42)
    network = Sequential(Linear(4, 4, rng), ReLU(), Linear(4, 4, rng))
    loss_fn = MSE_loss()
    opt = SGD(0.01)
    # optimizer = KristianSgd(learning_rate=0.1)

    x = np.array([[1, 2, 3, 4]])
    y = np.array([1, 1, 1, 1])

    before = [p.data.copy() for p in network.parameters()]

    epochs = 20

    for i in range(epochs):
        pred = network.forward(x)
        network.backward(loss_fn.backward())

        opt.step(network.parameters())

    for i, (p, b) in enumerate(zip(network.parameters(), before)):
        print(i, "endring:", np.linalg.norm(p.data - b))

    pred = network.forward(x)
    print(loss_fn.forward(pred, y))

    """train(
        nn=network,
        optimizer=optimizer,
        engine_name="frederik",
    )"""


if __name__ == "__main__":
    main()
