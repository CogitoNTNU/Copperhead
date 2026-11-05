from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from coppercore.shared import FloatArray, IntArray


class Batch(NamedTuple):
    states: FloatArray
    actions: IntArray
    rewards: FloatArray
    next_states: FloatArray
    terminated: FloatArray


class Buffer(ABC):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity

    @abstractmethod
    def add(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        terminated: bool,
    ) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def can_sample(self, batch_size: int) -> bool:
        return len(self) >= batch_size
