import math
from typing import Never, cast

from coppercore.model_driven_nn.components import (
    AvgPool2D,
    Component,
    Concat,
    Conv2D,
    Flatten,
    Linear,
    MaxPool2D,
    ReLU,
    Sequential,
)
from coppercore.model_driven_nn.construction.common import (
    non_negative_int,
    positive_int,
    validate_mapping,
)
from coppercore.shared import YamlMapping

type BuildResult = tuple[Component, int]


class Builder:
    def build(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None = None,
    ) -> BuildResult:
        validate_mapping(spec, "Component")
        component_type = spec.get("type")
        if not isinstance(component_type, str):
            raise TypeError("Component requires a string 'type'")

        method_name = "build_" + component_type
        builder = getattr(self, method_name, self.generic_build)
        return builder(spec, input_dim, shape_ctx)

    def _build_with_shape(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None,
    ) -> BuildResult:
        return self.build(spec, input_dim, shape_ctx)

    @staticmethod
    def generic_build(
        spec: YamlMapping,
        _input_dim: int,
        _shape_ctx: tuple[int, int, int] | None = None,
    ) -> Never:
        component_type = spec.get("type")

        if not isinstance(component_type, str):
            raise TypeError("Component requires a string 'type'")

        raise TypeError(f"Unsupported component type: {component_type!r}")

    @staticmethod
    def build_relu(
        _spec: YamlMapping,
        input_dim: int,
        _shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[ReLU, int]:
        return ReLU(), input_dim

    @staticmethod
    def build_flatten(
        _spec: YamlMapping,
        input_dim: int,
        _shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[Flatten, int]:
        return Flatten(), input_dim

    @staticmethod
    def build_linear(
        spec: YamlMapping,
        input_dim: int,
        _shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[Linear, int]:
        output_dim = positive_int(spec.get("output_dim"), "Linear output_dim")
        return Linear(input_dim, output_dim), output_dim

    @staticmethod
    def _parse_int_or_tuple(
        val: object,
        default: int | None,
        name: str,
        *,
        allow_zero: bool = False,
    ) -> tuple[int, int]:
        if val is None:
            if default is None:
                raise ValueError(f"{name} is required")
            return (default, default)
        checker = non_negative_int if allow_zero else positive_int
        if isinstance(val, int) and not isinstance(val, bool):
            v = checker(val, name)
            return (v, v)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return (checker(val[0], f"{name}[0]"), checker(val[1], f"{name}[1]"))
        raise TypeError(f"{name} must be an integer or a 2-element list of integers")

    @staticmethod
    def _resolve_spatial_shape(
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None,
        layer_name: str,
    ) -> tuple[tuple[int, int, int], int, int, int]:
        input_shape_spec = spec.get("input_shape")
        in_channels = (
            positive_int(spec["in_channels"], f"{layer_name} in_channels")
            if "in_channels" in spec
            else (shape_ctx[0] if shape_ctx is not None else 1)
        )

        input_shape: tuple[int, int, int] | None = None
        if input_shape_spec is not None:
            if not isinstance(input_shape_spec, (list, tuple)):
                raise TypeError(
                    f"{layer_name} input_shape must be a list or tuple of integers"
                )
            if len(input_shape_spec) == 3:
                input_shape = (
                    positive_int(input_shape_spec[0], f"{layer_name} input_shape[0]"),
                    positive_int(input_shape_spec[1], f"{layer_name} input_shape[1]"),
                    positive_int(input_shape_spec[2], f"{layer_name} input_shape[2]"),
                )
                in_channels = input_shape[0]
            elif len(input_shape_spec) == 2:
                input_shape = (
                    in_channels,
                    positive_int(input_shape_spec[0], f"{layer_name} input_shape[0]"),
                    positive_int(input_shape_spec[1], f"{layer_name} input_shape[1]"),
                )
            else:
                raise ValueError(
                    f"{layer_name} input_shape must contain 2 or 3 integers"
                )

            if input_shape[0] * input_shape[1] * input_shape[2] != input_dim:
                raise ValueError(
                    f"{layer_name} input_shape product ({input_shape[0] * input_shape[1] * input_shape[2]}) "
                    f"does not match input_dim ({input_dim})"
                )
        elif (
            shape_ctx is not None
            and shape_ctx[0] * shape_ctx[1] * shape_ctx[2] == input_dim
        ):
            input_shape = shape_ctx
            in_channels = input_shape[0]

        if input_shape is not None:
            in_h, in_w = input_shape[1], input_shape[2]
        else:
            spatial = input_dim // in_channels
            in_h = math.isqrt(spatial)
            if in_h * in_h == spatial:
                in_w = in_h
                input_shape = (in_channels, in_h, in_w)
            else:
                raise ValueError(
                    f"Cannot infer square spatial dimensions for input_dim={input_dim} "
                    f"with in_channels={in_channels}. Please specify input_shape."
                )

        return input_shape, in_channels, in_h, in_w

    def build_conv2d(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[Conv2D, int]:
        out_channels_raw = (
            spec.get("out_channels") or spec.get("output_dim") or spec.get("filters")
        )
        if out_channels_raw is None:
            raise ValueError(
                "Conv layer requires 'out_channels', 'output_dim', or 'filters'"
            )
        out_channels = positive_int(out_channels_raw, "Conv out_channels")

        input_shape, in_channels, in_h, in_w = self._resolve_spatial_shape(
            spec, input_dim, shape_ctx, "Conv"
        )

        kernel_size = self._parse_int_or_tuple(
            spec.get("kernel_size"), 3, "Conv kernel_size"
        )
        stride = self._parse_int_or_tuple(spec.get("stride"), 1, "Conv stride")
        padding = self._parse_int_or_tuple(
            spec.get("padding"), 0, "Conv padding", allow_zero=True
        )
        bias = spec.get("bias", True)
        if not isinstance(bias, bool):
            raise TypeError("Conv bias must be a boolean")

        k_h, k_w = kernel_size
        s_h, s_w = stride
        p_h, p_w = padding

        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Conv layer produces non-positive spatial dimensions ({out_h}, {out_w})"
            )

        output_dim = out_channels * out_h * out_w

        layer = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            input_shape=input_shape,
        )

        return layer, output_dim

    def _build_pool(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None,
        layer_cls: type[MaxPool2D | AvgPool2D],
        layer_name: str,
    ) -> tuple[MaxPool2D | AvgPool2D, int]:
        kernel_size = self._parse_int_or_tuple(
            spec.get("kernel_size"), 2, f"{layer_name} kernel_size"
        )
        stride = self._parse_int_or_tuple(
            spec.get("stride"), kernel_size[0], f"{layer_name} stride"
        )
        padding = self._parse_int_or_tuple(
            spec.get("padding"), 0, f"{layer_name} padding", allow_zero=True
        )

        input_shape, in_channels, in_h, in_w = self._resolve_spatial_shape(
            spec, input_dim, shape_ctx, layer_name
        )

        k_h, k_w = kernel_size
        s_h, s_w = stride
        p_h, p_w = padding

        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"{layer_name} layer produces non-positive spatial dimensions ({out_h}, {out_w})"
            )

        output_dim = in_channels * out_h * out_w
        layer = layer_cls(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_shape=input_shape,
        )
        return layer, output_dim

    def build_maxpool2d(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[MaxPool2D, int]:
        layer, out_dim = self._build_pool(
            spec, input_dim, shape_ctx, MaxPool2D, "MaxPool"
        )
        return cast(MaxPool2D, layer), out_dim

    def build_avgpool2d(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[AvgPool2D, int]:
        layer, out_dim = self._build_pool(
            spec, input_dim, shape_ctx, AvgPool2D, "AvgPool"
        )
        return cast(AvgPool2D, layer), out_dim

    def build_sequential(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[Sequential, int]:
        components: list[Component] = []
        dim = input_dim
        current_shape = shape_ctx

        for layer in self._children(spec, "layers"):
            component, dim = self.build(layer, dim, current_shape)
            components.append(component)

            if (
                hasattr(component, "output_shape")
                and component.output_shape is not None
            ):
                current_shape = component.output_shape
            elif isinstance(component, (Flatten, Linear)):
                current_shape = None

        return Sequential(components), dim

    def build_concat(
        self,
        spec: YamlMapping,
        input_dim: int,
        shape_ctx: tuple[int, int, int] | None = None,
    ) -> tuple[Concat, int]:
        branches: list[Component] = []
        output_dims: list[int] = []

        for path in self._children(spec, "paths"):
            branch, output_dim = self.build(path, input_dim, shape_ctx)
            branches.append(branch)
            output_dims.append(output_dim)

        return Concat(branches, output_dims), sum(output_dims)

    @staticmethod
    def _children(spec: YamlMapping, key: str) -> list[YamlMapping]:
        value = spec.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"'{key}' must be a nonempty list")

        children: list[YamlMapping] = []

        for child in value:
            children.append(validate_mapping(child, f"Child of '{key}'"))

        return children
