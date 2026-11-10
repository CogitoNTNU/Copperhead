from typing import cast

import numpy as np

from coppercore.shared import FloatArray


def to_2d_tuple(
    val: int | tuple[int, int] | list[int],
    name: str,
    *,
    positive: bool = False,
) -> tuple[int, int]:
    if isinstance(val, int) and not isinstance(val, bool):
        if positive and val <= 0:
            raise ValueError(f"{name} must be positive")
        if not positive and val < 0:
            raise ValueError(f"{name} must be non-negative")
        return (val, val)
    if isinstance(val, (tuple, list)) and len(val) == 2:
        for x in val:
            if not isinstance(x, int) or isinstance(x, bool):
                raise TypeError(f"{name} elements must be integers")
            if positive and x <= 0:
                raise ValueError(f"{name} elements must be positive integers")
            if not positive and x < 0:
                raise ValueError(f"{name} elements must be non-negative integers")
        return (val[0], val[1])
    raise TypeError(f"{name} must be an integer or a 2-tuple of integers")


def im2col(
    x_pad: FloatArray,
    k_h: int,
    k_w: int,
    stride_h: int,
    stride_w: int,
    h_out: int,
    w_out: int,
) -> FloatArray:
    n, c, _, _ = x_pad.shape
    cols = np.empty((c, k_h, k_w, n, h_out, w_out), dtype=x_pad.dtype)

    for i in range(k_h):
        i_max = i + stride_h * h_out
        for j in range(k_w):
            j_max = j + stride_w * w_out
            cols[:, i, j, :, :, :] = x_pad[
                :, :, i:i_max:stride_h, j:j_max:stride_w
            ].transpose(1, 0, 2, 3)

    return cast(FloatArray, cols.reshape(c * k_h * k_w, n * h_out * w_out))


def col2im(
    cols: FloatArray,
    x_pad_shape: tuple[int, int, int, int],
    k_h: int,
    k_w: int,
    stride_h: int,
    stride_w: int,
    h_out: int,
    w_out: int,
) -> FloatArray:
    n, c, _, _ = x_pad_shape
    cols_reshaped = cols.reshape(c, k_h, k_w, n, h_out, w_out)
    x_pad = np.zeros(x_pad_shape, dtype=cols.dtype)

    for i in range(k_h):
        i_max = i + stride_h * h_out
        for j in range(k_w):
            j_max = j + stride_w * w_out
            np.add.at(
                x_pad,
                (
                    slice(None),
                    slice(None),
                    slice(i, i_max, stride_h),
                    slice(j, j_max, stride_w),
                ),
                cols_reshaped[:, i, j, :, :, :].transpose(1, 0, 2, 3),
            )

    return x_pad
