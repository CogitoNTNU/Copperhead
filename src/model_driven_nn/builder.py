from typing import Never

from .components import Concat, Linear, ReLU, Sequential


class Builder:
    def build(self, layer, input_dim: int):
        method_name = "build_" + layer["type"]
        builder = getattr(self, method_name, self.generic_build)
        return builder(layer, input_dim)

    @staticmethod
    def generic_build(spec, _input_dim: int) -> Never:
        layer_type = spec["type"]
        raise ValueError(
            f"Unsupported component type: {layer_type!r}. "
            f"Expected a builder method named build_{layer_type}."
        )

    @staticmethod
    def build_linear(spec, input_dim: int) -> tuple[Linear, int]:
        output_dim = spec["output_dim"]
        return Linear(input_dim, output_dim), output_dim

    @staticmethod
    def build_relu(_spec, input_dim: int) -> tuple[ReLU, int]:
        return ReLU(), input_dim

    def build_sequential(self, spec, input_dim: int) -> tuple[Sequential, int]:
        components = []
        dim = input_dim

        for layer in spec["layers"]:
            component, dim = self.build(layer, dim)
            components.append(component)

        return Sequential(components), dim

    def build_concat(self, spec, input_dim: int) -> tuple[Concat, int]:
        branches = []
        output_dims = []

        for path in spec["paths"]:
            branch, output_dim = self.build(path, input_dim)
            branches.append(branch)
            output_dims.append(output_dim)

        return Concat(branches, output_dims), sum(output_dims)
