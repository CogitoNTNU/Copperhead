from packages.coppercollab.src.coppercollab.mnist_nn import (
    Sequential,
    Linear,
    ReLU,
    SGD,
)
import numpy as np
from collections import deque
import copy


# make network
rng = np.random.default_rng(seed=0)
# We send in 1 number defined as the state


class DQNAgent:
    def __init__(self, action_space, buffer_size, sample_size):
        self.eps = 1
        self.decay_eps = 0.99
        self.action_space = action_space
        self.replay_buffer = deque(maxlen=buffer_size)
        self.c = 100  # steps requiered to update self.c
        self.sample_size = sample_size
        self.buffer_size = buffer_size
        self.total_steps = 0  # to coun
        self.gamma = 0.95

        self.optimizer = SGD(learning_rate=0.001)
        self.network = Sequential(
            Linear(1, 128, rng),
            ReLU(),
            Linear(128, 128, rng),
            ReLU(),
            Linear(128, 4, rng),
        )
        # here we need deepcopy because sequential references other objects
        self.target_network = copy.deepcopy(self.network)

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
        return np.argmax(self.get_logit(state))

    def save_transition(self, state, action, reward, next_state):
        self.replay_buffer.appendleft((state, action, reward, next_state))

    def get_logit(self, state):
        logit = self.network.forward(state)
        return logit

    def gradient_step(self):
        # Sample minibatch
        # pass minibatch of next states through target net to get target Q
        # pass minibatch of states through self.network to get predicted values Q
        # compute loss and do backward + sgd
        pass

    def sample_minibatch(self, sample_size) -> np.ndarray | None:
        # guard against sampling when not filled
        if len(self.replay_buffer) < self.sample_size:
            return None

        # gererate minibatch_size amount of indices
        indices_array = rng.integers(len(self.replay_buffer), size=self.sample_size)
        replay_buffer_array = np.array(list(self.replay_buffer))
        return np.take(replay_buffer_array, indices_array, axis=0)

    def copy_weights(self):
        for target_param, network_param in zip(
            self.target_network.params(), self.network.params()
        ):
            # fine with .copy() here bc target_param.data is of ints, not list of other objects
            target_param.data = network_param.data.copy()
