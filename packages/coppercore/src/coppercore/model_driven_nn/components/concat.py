import operator
from collections.abc import Iterable
from functools import reduce

from coppercore.backend import get_array_module
from coppercore.shared import FloatArray

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

    def forward(
        self,
        x: FloatArray,
        *,
        cache: bool = True,
    ) -> FloatArray:
        outputs = [branch.forward(x, cache=cache) for branch in self.branches]
        branch_output_dims = [output.shape[-1] for output in outputs]

        if cache:
            self._branch_output_dims = branch_output_dims

        if (
            self.output_dims is not None
            and self._branch_output_dims != self.output_dims
        ):
            raise ValueError(
                "Concat branch output dimensions changed after construction"
            )
        xp = get_array_module(*outputs)
        return xp.concatenate(outputs, axis=-1)

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self._branch_output_dims is None:
            raise RuntimeError("Concat.backward() called before forward()")
        xp = get_array_module(d_y)
        splits = [
            sum(self._branch_output_dims[:i])
            for i in range(1, len(self._branch_output_dims))
        ]
        branch_gradients = xp.split(d_y, splits, axis=-1)
        input_gradients = [
            branch.backward(grad)
            for branch, grad in zip(self.branches, branch_gradients)
        ]
        if not input_gradients:
            return d_y
        return reduce(operator.add, input_gradients)

    def children(self) -> list[Component]:
        return self.branches
