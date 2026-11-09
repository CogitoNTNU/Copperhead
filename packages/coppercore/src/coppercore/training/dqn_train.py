from collections.abc import Callable

import numpy as np

from coppercore.model_driven_nn import Network
from coppercore.optimizer.optimizer import Optimizer
from coppercore.shared import FloatArray, IntArray
from coppercore.buffer import Buffer


def train_step(
    network: Network,
    optimizer: Optimizer,
    loss_fn: Callable,
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    network.zero_grad()
    out = network.forward(X)
    loss, loss_grad = loss_fn(out, Y)
    network.backward(loss_grad)
    optimizer.step()

    return loss


def dqn_mse(
    q_pred: FloatArray, target: tuple[IntArray, FloatArray]
) -> tuple[float, FloatArray]:
    actions, y = target
    N = len(actions)
    rows = np.arange(N)

    diff = q_pred[rows, actions] - y
    loss = float(np.mean(diff**2))

    grad = np.zeros_like(q_pred)
    grad[rows, actions] = 2.0 * diff / N  # derivate of loss function

    return loss, grad


def dqn_update(
    policy_net: Network,
    target_net: Network,
    optimizer: Optimizer,
    buffer: Buffer,
    batch_size: int,
    gamma: float,
    rng: np.random.Generator,
) -> float:
    state, action, reward, state_next, terminated = buffer.sample(batch_size, rng)

    a_next = policy_net.predict(state_next).argmax(axis=1)
    q_next = target_net.predict(state_next)[np.arange(batch_size), a_next]
    y = reward + gamma * q_next * (1.0 - terminated)  # reward or reward + gamma*q_next

    loss = train_step(
        policy_net, optimizer, dqn_mse, state, (action, y)
    )  # in-place update
    return loss
