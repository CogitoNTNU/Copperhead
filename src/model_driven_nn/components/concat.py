import numpy as np

from .base import Component


class Concat(Component):
    def __init__(self, branches, output_dims=None) -> None:
        self.branches = list(branches)
        self.output_dims = list(output_dims) if output_dims is not None else None
        self._branch_output_dims = None

    def forward(self, x):
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

    def backward(self, d_y):
        if self._branch_output_dims is None:
            raise RuntimeError("Concat.backward() called before forward()")
        splits = np.cumsum(self._branch_output_dims[:-1])
        branch_gradients = np.split(d_y, splits, axis=-1)
        input_gradients = [
            branch.backward(grad)
            for branch, grad in zip(self.branches, branch_gradients)
        ]
        return np.sum(input_gradients, axis=0)

    def params(self):
        return [parameter for branch in self.branches for parameter in branch.params()]
