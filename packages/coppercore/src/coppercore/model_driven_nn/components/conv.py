import math
from typing import Any, Self, cast

import numpy as np

from coppercore.backend import get_array_module, to_device
from coppercore.model_driven_nn.components.base import Layer
from coppercore.model_driven_nn.components.im2col import col2im, im2col, to_2d_tuple
from coppercore.model_driven_nn.parameter import Parameter
from coppercore.shared import FloatArray


class Conv2D(Layer):
    """2D Convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 0,
        bias: bool = True,
        input_shape: tuple[int, int, int] | tuple[int, int] | list[int] | None = None,
        seed: int = 0,
    ) -> None:
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2d_tuple(kernel_size, "kernel_size", positive=True)
        self.stride = to_2d_tuple(stride, "stride", positive=True)
        self.padding = to_2d_tuple(padding, "padding", positive=False)
        self.has_bias = bias

        self.input_shape: tuple[int, int, int] | None = None
        if input_shape is not None:
            if len(input_shape) == 3:
                self.input_shape = (input_shape[0], input_shape[1], input_shape[2])
            elif len(input_shape) == 2:
                self.input_shape = (in_channels, input_shape[0], input_shape[1])
            else:
                raise ValueError("input_shape must have 2 or 3 elements")

        rng = np.random.default_rng(seed)
        k_h, k_w = self.kernel_size
        scale = np.sqrt(2.0 / (in_channels * k_h * k_w))
        self.W = Parameter.from_data(
            rng.standard_normal((out_channels, in_channels, k_h, k_w)) * scale
        )
        self.b: Parameter | None = (
            Parameter.from_data(np.zeros(out_channels)) if bias else None
        )

        self._x: FloatArray | None = None
        self._orig_shape: tuple[int, ...] | None = None
        self._x_pad_shape: tuple[int, int, int, int] | None = None
        self._cols: FloatArray | None = None
        self._h_in: int = 0
        self._w_in: int = 0
        self._h_out: int = 0
        self._w_out: int = 0

    def to(self, device: str | None = None, dtype: Any = None) -> Self:
        super().to(device=device, dtype=dtype)
        if self._x is not None:
            self._x = to_device(self._x, device=device, dtype=dtype)
        if self._cols is not None:
            self._cols = to_device(self._cols, device=device, dtype=dtype)
        return self

    @property
    def output_shape(self) -> tuple[int, int, int] | None:
        if self.input_shape is not None:
            return self.compute_output_shape(self.input_shape[1], self.input_shape[2])
        if self._h_out > 0 and self._w_out > 0:
            return (self.out_channels, self._h_out, self._w_out)
        return None

    def compute_output_shape(self, in_h: int, in_w: int) -> tuple[int, int, int]:
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        h_out = (in_h + 2 * p_h - k_h) // s_h + 1
        w_out = (in_w + 2 * p_w - k_w) // s_w + 1

        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Calculated non-positive output dimensions ({h_out}, {w_out}) "
                f"for input ({in_h}, {in_w}) with kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}"
            )

        return (self.out_channels, h_out, w_out)

    def _prepare_input(self, x: FloatArray) -> tuple[FloatArray, int, int]:
        n = x.shape[0]
        if x.ndim == 4:
            _, _, h_in, w_in = x.shape
            return x, h_in, w_in

        if x.ndim == 3:
            h_in, w_in = x.shape[1], x.shape[2]
            return x.reshape(n, 1, h_in, w_in), h_in, w_in

        if x.ndim == 2:
            if self.input_shape is not None:
                c, h_in, w_in = self.input_shape
                return x.reshape(n, c, h_in, w_in), h_in, w_in

            total_features = x.shape[1]
            spatial = total_features // self.in_channels
            h_in = math.isqrt(spatial)
            if h_in * h_in == spatial:
                w_in = h_in
                return x.reshape(n, self.in_channels, h_in, w_in), h_in, w_in

            raise ValueError(
                f"Cannot infer 2D spatial dimensions for input of shape {x.shape} "
                f"and in_channels={self.in_channels}. Please specify input_shape."
            )

        raise ValueError(f"Expected 2D, 3D, or 4D input array, got {x.ndim}D")

    def forward(self, x: FloatArray, *, cache: bool = True) -> FloatArray:
        orig_shape = x.shape
        x_4d, h_in, w_in = self._prepare_input(x)
        n, _, _, _ = x_4d.shape

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

        xp = get_array_module(x_4d, self.W.data)

        if p_h > 0 or p_w > 0:
            x_pad = xp.pad(
                x_4d,
                ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
                mode="constant",
            )
        else:
            x_pad = x_4d

        cols = im2col(x_pad, k_h, k_w, s_h, s_w, h_out, w_out)
        w_row = self.W.data.reshape(self.out_channels, -1)

        out = w_row @ cols
        if self.b is not None:
            out = out + self.b.data[:, None]
        out = out.reshape(self.out_channels, n, h_out, w_out).transpose(1, 0, 2, 3)

        if cache:
            self._x = x
            self._orig_shape = orig_shape
            self._x_pad_shape = x_pad.shape
            self._cols = cols
            self._h_in = h_in
            self._w_in = w_in
            self._h_out = h_out
            self._w_out = w_out

        return out

    def backward(self, d_y: FloatArray) -> FloatArray:
        if (
            self._x is None
            or self._cols is None
            or self._x_pad_shape is None
            or self._orig_shape is None
        ):
            raise RuntimeError("Backward called before forward with cache=True")

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        n = self._orig_shape[0]

        if d_y.ndim == 2:
            d_y_4d = d_y.reshape(n, self.out_channels, self._h_out, self._w_out)
        else:
            d_y_4d = d_y

        if self.b is not None:
            self.b.grad += d_y_4d.sum(axis=(0, 2, 3))

        d_y_trans = d_y_4d.transpose(1, 0, 2, 3).reshape(
            self.out_channels, n * self._h_out * self._w_out
        )
        d_w_row = d_y_trans @ self._cols.T
        self.W.grad += d_w_row.reshape(self.W.data.shape)

        w_row = self.W.data.reshape(self.out_channels, -1)
        d_cols = w_row.T @ d_y_trans

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

        if len(self._orig_shape) == 2:
            return cast(FloatArray, d_x.reshape(self._orig_shape))

        return cast(FloatArray, d_x)

    def params(self) -> list[Parameter]:
        return [self.W, self.b] if self.b is not None else [self.W]

    def zero_grad(self) -> None:
        for param in self.params():
            param.zero_grad()
