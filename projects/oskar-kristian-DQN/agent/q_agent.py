import numpy as np


class QAgent:
    def __init__(self, n_states, n_actions, seed=0, alpha=0.1, gamma=0.99):
        self.rng = np.random.default_rng(seed)
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.alpha = alpha
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        self.q = np.zeros((n_states, n_actions))

    @property
    def parals(self) -> dict:
        return {"q": self.q}

    def update_qtable(self, state, action, reward, next_state, terminated):
        td_target = reward + self.gamma * np.max(self.q[next_state]) * (1 - terminated)
        self.q[state][action] = self.q[state][action] + self.alpha * (
            td_target - self.q[state][action]
        )

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        num = self.rng.random()
        if not greedy and num < self.epsilon:
            return self.rng.integers(self.n_actions)
        else:
            return np.argmax(self.q[state])

    def observe(self, state, action, reward, next_state, terminated) -> None:
        self.update_qtable(state, action, reward, next_state, terminated)
        if terminated:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
