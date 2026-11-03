from coppercollab.mnist_nn import Sequential, Linear, ReLU
from coppercore.optimizer.adam import ADAM
import numpy as np
from collections import deque
import copy


# make network
rng = np.random.default_rng(seed=0)
# We send in 1 number defined as the state


class DQNAgent:
    def __init__(self, state_dim, action_space, buffer_size, sample_size):
        self.eps = 1
        self.decay_eps = 0.99
        self.action_space = action_space
        self.replay_buffer = deque(maxlen=buffer_size)
        self.c = 100  # steps requiered to update self.c
        self.sample_size = sample_size
        self.buffer_size = buffer_size
        self.total_steps = 0  # to coun
        self.gamma = 0.95

        self.optimizer = ADAM(learning_rate=0.001)
        self.network = Sequential(
            Linear(state_dim, 128, rng),
            ReLU(),
            Linear(128, 128, rng),
            ReLU(),
            Linear(128, action_space, rng),
        )
        self.optimizer.parameters = self.network.params()
        # here we need deepcopy because sequential references other objects
        self.target_network = copy.deepcopy(self.network)

    """Agent procedure"""

    def act(self, state, greedy: bool = False):  # eps greedy exploration
        num = rng.random()
        if not greedy and num < self.eps:
            return rng.integers(self.action_space)
        else:
            return self.get_action(state)

    def observe(self, state, action, reward, next_state, terminated):
        self.total_steps += 1

        self.save_transition(state, action, reward, next_state, terminated)
        self.gradient_step()

        if self.total_steps % self.c == 0:
            self.copy_weights()

        if terminated:
            self.eps = max(0.01, self.eps * self.decay_eps)
        # decay exploration epsilon
        pass

    """Helper functions"""

    def get_action(self, state):
        return np.argmax(self.network.forward(state))

    def save_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.appendleft((state, action, reward, next_state, done))

    def get_logit(self, state):
        logit = self.network.forward(state)
        return logit

    def gradient_step(self):
        # Sample minibatch
        minibatch = self.sample_minibatch()
        if minibatch is None:
            return

        states, actions, rewards, next_states, dones = zip(*minibatch)  # unzip tuples
        # reshape to (sample_size,1) for matmul
        # others dont need to because they get broadcasted
        states = np.array(states, dtype=np.float32).reshape(len(minibatch), -1)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32).reshape(
            len(minibatch), -1
        )
        dones = np.array(dones, dtype=np.float32)

        max_next_q = np.max(self.target_network.forward(next_states), axis=1)
        td_target = rewards + self.gamma * max_next_q * (1 - dones)

        Q = self.network.forward(states)  # (32,4)

        target_Q = Q.copy()
        target_Q[np.arange(len(minibatch)), actions] = td_target
        # compute loss and do backward + sgd
        loss, grad = self.mse_loss(Q, target_Q)
        self.network.backward(grad)
        self.optimizer.step()

    def sample_minibatch(self) -> np.ndarray | None:
        # guard against sampling when not filled
        if len(self.replay_buffer) < self.sample_size:
            return None
        # gererate minibatch_size amount of indices
        indices = rng.integers(len(self.replay_buffer), size=self.sample_size)

        minibatch = [self.replay_buffer[i] for i in indices]
        return minibatch  # len = sample_size, of 5 tuples, state,action,reward,next_state,done

    def copy_weights(self):
        for target_param, network_param in zip(
            self.target_network.params(), self.network.params()
        ):
            # fine with .copy() here bc target_param.data is of ints, not list of other objects
            target_param.data = network_param.data.copy()

    def mse_loss(self, Q, Q_target):
        return np.mean((Q - Q_target) ** 2), 2 * (Q - Q_target) / len(Q)
