import numpy as np

from coppercore.dqn.base_agent import BaseAgent
from coppercore.dqn.snake_env import SnakeEnv
from coppercore.training.dqn_harness import train
from coppercore.dqn.snake_render import RenderedSnakeEnv


def make_snake_env():
    return SnakeEnv(width=10, height=10, seed=0, max_steps_no_food=200)


class TabularQAgent(BaseAgent):
    def __init__(self, n_states: int, n_actions: int, seed: int = 0):
        self.n_state = n_states
        self.n_actions = n_actions

        self.rng = np.random.default_rng(seed)
        self.learning_rate = 0.1
        self.gamma = 0.90

        # Exploration parameters
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.decay_steps = 25_000
        self.epsilon = self.max_epsilon

        self.steps = 0

        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.params = {"Q": self.Q}

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if greedy or self.rng.random() >= self.epsilon:
            # exploit and choose the best action
            q = self.Q[obs]
            return int(self.rng.choice(np.flatnonzero(q == q.max())))
        else:
            # explore other actions
            # TODO: remove greedy action from selection
            return self.rng.integers(self.n_actions)

    def observe(self, obs, action, reward, next_obs, terminated) -> dict | None:
        target = reward if terminated else reward + self.gamma * self.Q[next_obs].max()
        td_error = target - self.Q[obs, action]
        self.Q[obs, action] += self.learning_rate * td_error

        self.steps += 1
        frac = min(1.0, self.steps / self.decay_steps)
        self.epsilon = self.max_epsilon + frac * (self.min_epsilon - self.max_epsilon)

        return None  # a returned dict will be logged in WandB


env = make_snake_env()
agent = TabularQAgent(n_states=env.n_states, n_actions=env.n_actions, seed=42)

train(
    agent,
    make_env=make_snake_env,
    n_steps=100_000,
    eval_every=500,
    group="fred_test",
    name="Snake-Q-table",
    make_video_env=lambda: RenderedSnakeEnv(
        width=10, height=10, max_steps_no_food=200, render_mode="rgb_array"
    ),
)
