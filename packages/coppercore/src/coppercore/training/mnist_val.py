from copperhead.data import load_mnist
from copperhead.model_driven_nn import Network


def validate(network: Network, savepath="data/basic_nn_weights.npz", seed=0):
    _, _, X_val, y_val = load_mnist(seed=seed)

    # network.load(savepath)
    #
    # logits = network.forward(X_val)
    # acc = accuracy(logits, y_val)
    # print(f"val accuracy {acc:.4f}")
    # return acc


if __name__ == "__main__":
    pass
    # validate()
