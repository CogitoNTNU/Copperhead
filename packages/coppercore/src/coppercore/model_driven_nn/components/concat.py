from collections.abc import Iterable

import numpy as np

from coppercore.shared import FloatArray, YamlMapping

from ..parameter import Parameter
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

        if self.output_dims is not None and branch_output_dims != self.output_dims:
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

    def children(self) -> list[Component]:
        return self.branches

    def to_config(self) -> YamlMapping:
        return {
            "type": "concat",
            "paths": [branch.to_config() for branch in self.branches],
        }

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        named_params = []
        for index, branch in enumerate(self.branches):
            child_prefix = f"{prefix}.paths.{index}" if prefix else f"paths.{index}"
            named_params.extend(branch.named_parameters(prefix=child_prefix))
        return named_params
