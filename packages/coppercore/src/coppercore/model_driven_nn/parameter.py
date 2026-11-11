# parameter.py

from dataclasses import dataclass
from typing import Any

from coppercore.backend import get_array_module, to_device
from coppercore.shared import FloatArray


@dataclass
class Parameter:
    data: FloatArray
    grad: FloatArray

    @classmethod
    def from_data(cls, data: FloatArray) -> "Parameter":
        xp = get_array_module(data)
        return cls(
            data=data,
            grad=xp.zeros_like(data),
        )

    def zero_grad(self) -> None:
        self.grad.fill(0)

    def to(self, device: str | None = None, dtype: Any = None) -> "Parameter":
        self.data = to_device(self.data, device=device, dtype=dtype)
        self.grad = to_device(self.grad, device=device, dtype=dtype)
        return self
