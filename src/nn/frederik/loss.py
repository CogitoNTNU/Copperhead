import numpy as np


class MSE_loss:
    def forward(self, pred, target, mask=None):
        self.diff = pred - target
        self.mask = np.ones_like(pred) if mask is None else mask
        self.n = pred.shape[0]

        return 1 / 2 * np.sum((self.diff * self.mask) ** 2) / self.n

    def backward(self):
        return self.diff * self.mask / self.n
