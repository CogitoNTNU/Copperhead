from packages.coppercollab.src.coppercollab.mnist_nn import Sequential, Linear, ReLU
import numpy as np
from collections import deque


# make network
rng = np.random.default_rng(seed=0)
# We send in 1 number defined as the state
target_network = Sequential(
    Linear(1, 128, rng), ReLU(), Linear(64, 128, rng), ReLU(), Linear(128, 4, rng)
)


class DQNAgent:
    def __init__(self, action_space, buffer_size, sample_size):
        self.eps = 1
        self.decay_eps = 0.99
        self.action_space = action_space
        self.replay_buffer = deque(maxlen=buffer_size)
        self.c = 100  # steps requiered to update self.c
        self.sample_size = sample_size
        self.buffer_size = buffer_size

    """Agent procedure"""

    def act(self, state):  # eps greedy exploration
        num = rng.random()
        if num < self.eps:
            return rng.integers(self.action_space)
        else:
            return self.get_action(state)

    def observe(self):
        pass

    """Helper functions"""

    def get_action(self, state):
        return np.argmax(self.get_target(state))

    def save_transition(self, state, action, reward, next_state):
        self.replay_buffer.appendleft((state, action, reward, next_state))

    def get_target(self, state, minibatch):
        target = target_network.forward(state)
        return target

    def gradient_step(self):
        pass

    def sample_minibatch(self, sample_size) -> np.ndarray:
        # gererate minibatch_size amount of indices
        indices_array = rng.integers(len(self.replay_buffer), size=self.sample_size)
        replay_buffer_array = np.array(list(self.replay_buffer))
        return np.take(replay_buffer_array, indices_array)

    def copy_weights(self):
        pass
