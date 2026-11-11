from collections.abc import Callable
from typing import Any

import numpy as np

from coppercore.backend import get_array_module, to_device
from coppercore.buffer import Buffer
from coppercore.model_driven_nn import Network
from coppercore.optimizer.optimizer import Optimizer
from coppercore.shared import FloatArray, IntArray


def train_step(
    network: Network,
    optimizer: Optimizer,
    loss_fn: Callable,
    X: np.ndarray,
    Y: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> Any:
    network.zero_grad()
    out = network.forward(X)
    loss, loss_grad = loss_fn(out, Y)
    network.backward(loss_grad)
    optimizer.step()

    return loss


def dqn_mse(
    q_pred: FloatArray, target: tuple[IntArray, FloatArray]
) -> tuple[Any, FloatArray]:
    actions, y = target
    xp = get_array_module(q_pred, actions, y)
    N = len(actions)
    rows = xp.arange(N)

    diff = q_pred[rows, actions] - y
    loss = xp.mean(diff**2)

    grad = xp.zeros_like(q_pred)
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
    device: str | None = None,
    dtype: Any = None,
) -> float:
    state, action, reward, state_next, terminated = buffer.sample(batch_size, rng)

    params = policy_net.params()
    if params:
        p_xp = get_array_module(params[0].data)
        if device is None:
            device = "cpu" if p_xp is np else "gpu"
        if dtype is None:
            dtype = params[0].data.dtype

    if device == "gpu":
        state_dev = to_device(state, device="gpu", dtype=dtype)
        state_next_dev = to_device(state_next, device="gpu", dtype=dtype)
        action_dev = to_device(action, device="gpu")
        reward_dev = to_device(reward, device="gpu", dtype=dtype)
        terminated_dev = to_device(terminated, device="gpu", dtype=dtype)
    elif dtype is not None:
        state_dev = state.astype(dtype) if state.dtype != dtype else state
        state_next_dev = (
            state_next.astype(dtype) if state_next.dtype != dtype else state_next
        )
        action_dev = action
        reward_dev = reward.astype(dtype) if reward.dtype != dtype else reward
        terminated_dev = (
            terminated.astype(dtype) if terminated.dtype != dtype else terminated
        )
    else:
        state_dev, state_next_dev, action_dev, reward_dev, terminated_dev = (
            state,
            state_next,
            action,
            reward,
            terminated,
        )

    pred_next = policy_net.predict(state_next_dev)
    xp = get_array_module(pred_next)
    a_next = pred_next.argmax(axis=1)
    target_pred = target_net.predict(state_next_dev)
    q_next = target_pred[xp.arange(batch_size), a_next]
    y = reward_dev + gamma * q_next * (1.0 - terminated_dev)

    loss = train_step(
        policy_net, optimizer, dqn_mse, state_dev, (action_dev, y)
    )  # in-place update
    return loss
