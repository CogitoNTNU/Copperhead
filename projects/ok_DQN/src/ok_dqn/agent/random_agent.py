import numpy as np


class RandomAgent:
    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)
        self.params = {}

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        return self.rng.integers(self.action_space.n)

    def observe(self, obs, action, reward, next_obs, terminated) -> dict | None:
        return {"last_action": action}
