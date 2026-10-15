import numpy as np


def softmax_cross_entropy(logits, y_true):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    n = logits.shape[0]
    log_likelihood = -np.log(probs[np.arange(n), y_true] + 1e-12)
    loss = log_likelihood.mean()

    grad = probs.copy()
    grad[np.arange(n), y_true] -= 1
    grad /= n

    return loss, grad
