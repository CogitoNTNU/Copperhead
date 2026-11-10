from coppercore.model_driven_nn.components.base import Layer
from coppercore.shared import FloatArray


class Flatten(Layer):
    """Flattens multidimensional inputs into 2D batch arrays (N, features)."""

    def __init__(self) -> None:
        self._orig_shape: tuple[int, ...] | None = None

    def forward(
        self,
        x: FloatArray,
        *,
        cache: bool = True,
    ) -> FloatArray:
        if cache:
            self._orig_shape = x.shape
        if x.ndim <= 2:
            return x
        return x.reshape(x.shape[0], -1)

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self._orig_shape is None:
            raise RuntimeError("Flatten.backward() called before forward()")
        if len(self._orig_shape) <= 2:
            return d_y
        return d_y.reshape(self._orig_shape)
