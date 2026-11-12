import math
from typing import Any, Self, cast

from coppercore.backend import get_array_module, to_device
from coppercore.model_driven_nn.components.base import Layer
from coppercore.model_driven_nn.components.im2col import col2im, im2col, to_2d_tuple
from coppercore.shared import FloatArray, IntArray, YamlMapping


class _Pool2DBase(Layer):
    def __init__(
        self,
        kernel_size: int | tuple[int, int] | list[int] = 2,
        stride: int | tuple[int, int] | list[int] | None = None,
        padding: int | tuple[int, int] | list[int] = 0,
        input_shape: tuple[int, int, int] | tuple[int, int] | list[int] | None = None,
    ) -> None:
        self.kernel_size = to_2d_tuple(kernel_size, "kernel_size", positive=True)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = to_2d_tuple(stride, "stride", positive=True)
        self.padding = to_2d_tuple(padding, "padding", positive=False)

        self.input_shape: tuple[int, int, int] | None = None
        if input_shape is not None:
            if len(input_shape) == 3:
                self.input_shape = (input_shape[0], input_shape[1], input_shape[2])
            elif len(input_shape) == 2:
                self.input_shape = (1, input_shape[0], input_shape[1])
            else:
                raise ValueError("input_shape must have 2 or 3 elements")

        self._orig_shape: tuple[int, ...] | None = None
        self._x_pad_shape: tuple[int, int, int, int] | None = None
        self._c_in: int = 0
        self._h_in: int = 0
        self._w_in: int = 0
        self._h_out: int = 0
        self._w_out: int = 0

    @property
    def output_shape(self) -> tuple[int, int, int] | None:
        if self.input_shape is not None:
            return self.compute_output_shape(
                self.input_shape[0], self.input_shape[1], self.input_shape[2]
            )
        if self._c_in > 0 and self._h_out > 0 and self._w_out > 0:
            return (self._c_in, self._h_out, self._w_out)
        return None

    def compute_output_shape(
        self, in_channels: int, in_h: int, in_w: int
    ) -> tuple[int, int, int]:
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        h_out = (in_h + 2 * p_h - k_h) // s_h + 1
        w_out = (in_w + 2 * p_w - k_w) // s_w + 1

        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Calculated non-positive output dimensions ({h_out}, {w_out}) "
                f"for input ({in_channels}, {in_h}, {in_w}) with kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}"
            )

        return (in_channels, h_out, w_out)

    def _prepare_input(self, x: FloatArray) -> tuple[FloatArray, int, int, int]:
        n = x.shape[0]
        if x.ndim == 4:
            _, c_in, h_in, w_in = x.shape
            return x, c_in, h_in, w_in

        if x.ndim == 3:
            h_in, w_in = x.shape[1], x.shape[2]
            return x.reshape(n, 1, h_in, w_in), 1, h_in, w_in

        if x.ndim == 2:
            if self.input_shape is not None:
                c_in, h_in, w_in = self.input_shape
                return x.reshape(n, c_in, h_in, w_in), c_in, h_in, w_in

            total_features = x.shape[1]
            h_in = math.isqrt(total_features)
            if h_in * h_in == total_features:
                w_in = h_in
                return x.reshape(n, 1, h_in, w_in), 1, h_in, w_in

            raise ValueError(
                f"Cannot infer 2D spatial dimensions for input of shape {x.shape}. "
                "Please specify input_shape."
            )

        raise ValueError(f"Expected 2D, 3D, or 4D input array, got {x.ndim}D")

    def _base_config(self, pool_type: str) -> YamlMapping:
        config: YamlMapping = {
            "type": pool_type,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": list(self.padding),
        }
        if self.input_shape is not None:
            config["input_shape"] = list(self.input_shape)
        return config


class MaxPool2D(_Pool2DBase):
    """2D Max Pooling layer."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | list[int] = 2,
        stride: int | tuple[int, int] | list[int] | None = None,
        padding: int | tuple[int, int] | list[int] = 0,
        input_shape: tuple[int, int, int] | tuple[int, int] | list[int] | None = None,
    ) -> None:
        super().__init__(kernel_size, stride, padding, input_shape)
        self._max_idx: IntArray | None = None

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        super().to(device=device, dtype=dtype)
        if self._max_idx is not None:
            self._max_idx = to_device(self._max_idx, device=device)
        return self

    def forward(self, x: FloatArray, *, cache: bool = True) -> FloatArray:
        orig_shape = x.shape
        x_4d, c_in, h_in, w_in = self._prepare_input(x)
        n = x_4d.shape[0]

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        h_out = (h_in + 2 * p_h - k_h) // s_h + 1
        w_out = (w_in + 2 * p_w - k_w) // s_w + 1

        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Output dimensions ({h_out}, {w_out}) must be positive. "
                f"Input spatial size: ({h_in}, {w_in}), kernel: ({k_h}, {k_w}), "
                f"stride: ({s_h}, {s_w}), padding: ({p_h}, {p_w})"
            )

        x_reshaped = x_4d.reshape(n * c_in, 1, h_in, w_in)
        xp = get_array_module(x_reshaped)
        if p_h > 0 or p_w > 0:
            x_pad = xp.pad(
                x_reshaped,
                ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
                mode="constant",
                constant_values=-xp.inf,
            )
        else:
            x_pad = x_reshaped

        cols = im2col(x_pad, k_h, k_w, s_h, s_w, h_out, w_out)
        max_idx = xp.argmax(cols, axis=0)
        max_vals = cols[max_idx, xp.arange(cols.shape[1])]
        out = max_vals.reshape(n, c_in, h_out, w_out)

        if cache:
            self._orig_shape = orig_shape
            self._x_pad_shape = x_pad.shape
            self._c_in = c_in
            self._h_in = h_in
            self._w_in = w_in
            self._h_out = h_out
            self._w_out = w_out
            self._max_idx = max_idx

        return out

    def backward(self, d_y: FloatArray) -> FloatArray:
        if (
            self._max_idx is None
            or self._x_pad_shape is None
            or self._orig_shape is None
        ):
            raise RuntimeError("MaxPool2D.backward() called before forward()")

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        n = self._orig_shape[0]

        if d_y.ndim == 2:
            d_y_4d = d_y.reshape(n, self._c_in, self._h_out, self._w_out)
        else:
            d_y_4d = d_y

        xp = get_array_module(d_y, self._max_idx)
        num_windows = n * self._c_in * self._h_out * self._w_out
        d_cols = xp.zeros((k_h * k_w, num_windows), dtype=d_y.dtype)
        d_cols[self._max_idx, xp.arange(num_windows)] = d_y_4d.reshape(-1)

        d_x_pad = col2im(
            d_cols,
            self._x_pad_shape,
            k_h,
            k_w,
            s_h,
            s_w,
            self._h_out,
            self._w_out,
        )

        if p_h > 0 or p_w > 0:
            d_x = d_x_pad[:, :, p_h : p_h + self._h_in, p_w : p_w + self._w_in]
        else:
            d_x = d_x_pad

        d_x_4d = d_x.reshape(n, self._c_in, self._h_in, self._w_in)
        if len(self._orig_shape) == 2:
            return cast(FloatArray, d_x_4d.reshape(self._orig_shape))

        return cast(FloatArray, d_x_4d)

    def to_config(self) -> YamlMapping:
        return self._base_config("maxpool2d")


class AvgPool2D(_Pool2DBase):
    """2D Average Pooling layer."""

    def forward(self, x: FloatArray, *, cache: bool = True) -> FloatArray:
        orig_shape = x.shape
        x_4d, c_in, h_in, w_in = self._prepare_input(x)
        n = x_4d.shape[0]

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        h_out = (h_in + 2 * p_h - k_h) // s_h + 1
        w_out = (w_in + 2 * p_w - k_w) // s_w + 1

        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Output dimensions ({h_out}, {w_out}) must be positive. "
                f"Input spatial size: ({h_in}, {w_in}), kernel: ({k_h}, {k_w}), "
                f"stride: ({s_h}, {s_w}), padding: ({p_h}, {p_w})"
            )

        x_reshaped = x_4d.reshape(n * c_in, 1, h_in, w_in)
        xp = get_array_module(x_reshaped)
        if p_h > 0 or p_w > 0:
            x_pad = xp.pad(
                x_reshaped,
                ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
                mode="constant",
                constant_values=0.0,
            )
        else:
            x_pad = x_reshaped

        cols = im2col(x_pad, k_h, k_w, s_h, s_w, h_out, w_out)
        mean_vals = xp.mean(cols, axis=0)
        out = mean_vals.reshape(n, c_in, h_out, w_out)

        if cache:
            self._orig_shape = orig_shape
            self._x_pad_shape = x_pad.shape
            self._c_in = c_in
            self._h_in = h_in
            self._w_in = w_in
            self._h_out = h_out
            self._w_out = w_out

        return out

    def backward(self, d_y: FloatArray) -> FloatArray:
        if self._x_pad_shape is None or self._orig_shape is None:
            raise RuntimeError("AvgPool2D.backward() called before forward()")

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        n = self._orig_shape[0]

        if d_y.ndim == 2:
            d_y_4d = d_y.reshape(n, self._c_in, self._h_out, self._w_out)
        else:
            d_y_4d = d_y

        xp = get_array_module(d_y)
        grad_per_elem = d_y_4d.reshape(-1) / (k_h * k_w)
        d_cols = xp.tile(grad_per_elem, (k_h * k_w, 1))

        d_x_pad = col2im(
            d_cols,
            self._x_pad_shape,
            k_h,
            k_w,
            s_h,
            s_w,
            self._h_out,
            self._w_out,
        )

        if p_h > 0 or p_w > 0:
            d_x = d_x_pad[:, :, p_h : p_h + self._h_in, p_w : p_w + self._w_in]
        else:
            d_x = d_x_pad

        d_x_4d = d_x.reshape(n, self._c_in, self._h_in, self._w_in)
        if len(self._orig_shape) == 2:
            return cast(FloatArray, d_x_4d.reshape(self._orig_shape))

        return cast(FloatArray, d_x_4d)

    def to_config(self) -> YamlMapping:
        return self._base_config("avgpool2d")
