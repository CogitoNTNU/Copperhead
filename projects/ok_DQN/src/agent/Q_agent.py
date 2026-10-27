import numpy as np


class QAgent:
    def __init__(self, n_states, n_actions, seed=0, alpha=0.05, gamma=0.99):
        self.rng = np.random.default_rng(seed)
        self.epsilon = 1
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.alpha = alpha
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        self.q = np.zeros((n_states, n_actions))
        self.td_error_ema = 0

    @property
    def params(self) -> dict:
        return {"q": self.q}

    def update_qtable(self, state, action, reward, next_state, terminated) -> float:
        td_target = reward + self.gamma * np.max(self.q[next_state]) * (1 - terminated)
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        return float(abs(td_error))

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        num = self.rng.random()
        if not greedy and num < self.epsilon:
            return self.rng.integers(self.n_actions)
        else:
            return np.argmax(self.q[state])

    def observe(self, state, action, reward, next_state, terminated) -> None | dict:
        td_error = self.update_qtable(state, action, reward, next_state, terminated)
        if terminated:  # or truncated
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.td_error_ema = 0.99 * self.td_error_ema + 0.01 * td_error
        # for logging
        return {"td_error_ma": self.td_error_ema, "epsilon": self.epsilon}
