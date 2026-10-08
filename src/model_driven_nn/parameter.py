# parameter.py

from dataclasses import dataclass

import numpy as np

from shared.types import FloatArray


@dataclass
class Parameter:
    data: FloatArray
    grad: FloatArray

    @classmethod
    def from_data(cls, data: FloatArray) -> "Parameter":
        return cls(
            data=data,
            grad=np.zeros_like(data),
        )

    def zero_grad(self) -> None:
        self.grad.fill(0)
