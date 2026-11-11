import numpy as np

from .buffer import Batch, Buffer


class ReplayBuffer(Buffer):
    """A ring-buffer of numpy arrays, that works like a deque.
    Made for a Snake DQN.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        obs_dtype: np.dtype = np.uint8,
    ) -> None:
        super().__init__(capacity)
        self.states = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, terminated) -> None:
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if not self.can_sample(batch_size):
            raise RuntimeError(
                f"Cannot sample {batch_size} transitions, buffer only holds {len(self)}"
            )
        idx = rng.integers(0, self.size, size=batch_size)
        states = (
            self.states[idx].astype(np.float32)
            if self.states.dtype != np.float32
            else self.states[idx]
        )
        next_states = (
            self.next_states[idx].astype(np.float32)
            if self.next_states.dtype != np.float32
            else self.next_states[idx]
        )
        return Batch(
            states,
            self.actions[idx],
            self.rewards[idx],
            next_states,
            self.terminated[idx],
        )

    def __len__(self) -> int:
        return self.size
