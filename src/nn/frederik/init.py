import numpy as np


def default_init(rng: np.random.Generator, shape, lim, dtype=np.float32):
    return rng.uniform(-lim, lim, size=shape).astype(dtype)
