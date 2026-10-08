from collections import defaultdict

import gymnasium as gym
import numpy as np


class CartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters: how much exploration are we going to do
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def act(self, observation: tuple[float, float, float, float], greedy: bool):

        if greedy:
            return int(np.argmax(self.q_values[observation]))

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return int(np.argmax(self.q_values[observation]))

    def observe(
        self,
        obs: tuple[float, float, float, float],
        action: int,
        reward: float,
        next_obs: tuple[float, float, float, float],
        terminated: bool,
    ):

        next_obs = tuple(next_obs)
        obs = tuple(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value


        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    #TODO discretize
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path):
        return None

    def load(self, path):
        return None
