from pathlib import Path
import numpy as np


cache_path = Path(__file__).resolve().parents[2] / 'data' / 'mnist.npz'

def download_and_cache():
    from sklearn.datasets import fetch_openml

    X,y = fetch_openml(
        'mnist_784', version = 1, return_X_y = True, as_frame = False, parser = 'auto'
    )

    y = y.astype(np.int64)

    cache_path.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(cache_path, X = X, y = y)
    return X,y


def load_mnist(seed = 0, val_fraction = 0.1):

    if cache_path.exists():
       cached = np.load(cache_path)
       X, y = cached['X'], cached['y']
    else:
       X,y = download_and_cache()


    X = X.astype(np.float64) / 255.0

    rng = np.random.default_rng(seed)

    perm = rng.permutation(len(X))

    X,y = X[perm], y[perm]

    n_val = int(len(X) * val_fraction)

    X_val, y_val = X[:n_val], y[:n_val] 
    X_train, y_train = X[n_val:], y[n_val:]

    return X_train, y_train, X_val, y_val



if __name__ == '__main__':
    X_train, y_train, X_val, y_val = load_mnist()