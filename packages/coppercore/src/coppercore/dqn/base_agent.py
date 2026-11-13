import numpy as np


class BaseAgent:
    """An agent that plays randomly.
    Anything put in self.params (a dict of np.ndarray) is logged to WandB by train().
    """

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.params = {}

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def act_batch(self, observations: np.ndarray, greedy: bool = True) -> np.ndarray:
        batch_size = len(observations)
        if batch_size == 0:
            return np.empty(0, dtype=np.int64)
        return self.rng.integers(self.n_actions, size=batch_size, dtype=np.int64)

    def observe(self, obs, action, reward, next_obs, terminated) -> dict | None:
        return None  # a returned dict will be logged in WandB
