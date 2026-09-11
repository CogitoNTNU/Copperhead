from collections.abc import Iterable

import numpy as np

from ..types import FloatArray, ParameterPair
from .base import Component, Structure


class Concat(Structure):
    def __init__(
        self, branches: Iterable[Component], output_dims: list[int] | None = None
    ) -> None:
        self.branches: list[Component] = list(branches)
        self.output_dims: list[int] | None = (
            list(output_dims) if output_dims is not None else None
        )
        self._branch_output_dims: list[int] | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        outputs = [branch.forward(x) for branch in self.branches]
        self._branch_output_dims = [output.shape[-1] for output in outputs]
        if (
            self.output_dims is not None
            and self._branch_output_dims != self.output_dims
        ):
            raise ValueError(
                "Concat branch output dimensions changed after construction"
            )
        return np.concatenate(outputs, axis=-1)

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self._branch_output_dims is None:
            raise RuntimeError("Concat.backward() called before forward()")
        splits = np.cumsum(self._branch_output_dims[:-1])
        branch_gradients = np.split(d_y, splits, axis=-1)
        input_gradients = [
            branch.backward(grad)
            for branch, grad in zip(self.branches, branch_gradients)
        ]
        return np.sum(input_gradients, axis=0)

    def params(self) -> list[ParameterPair]:
        return [parameter for branch in self.branches for parameter in branch.params()]
