from collections.abc import Mapping
from typing import Never

from .components import Component, Concat, Linear, ReLU, Sequential

type ArchitectureSpec = Mapping[str, object]
type BuildResult = tuple[Component, int]


class Builder:
    def build(self, spec: ArchitectureSpec, input_dim: int) -> BuildResult:
        self._validate_mapping(spec, "Component")
        component_type = spec.get("type")
        if not isinstance(component_type, str):
            raise TypeError("Component requires a string 'type'")

        method_name = "build_" + component_type
        builder = getattr(self, method_name, self.generic_build)
        return builder(spec, input_dim)

    @staticmethod
    def generic_build(spec: ArchitectureSpec, _input_dim: int) -> Never:
        component_type = spec.get("type")

        if not isinstance(component_type, str):
            raise TypeError("Component requires a string 'type'")

        raise TypeError(f"Unsupported component type: {component_type!r}")

    @staticmethod
    def build_relu(_spec: ArchitectureSpec, input_dim: int) -> tuple[ReLU, int]:
        return ReLU(), input_dim

    def build_linear(
        self, spec: ArchitectureSpec, input_dim: int
    ) -> tuple[Linear, int]:
        output_dim = self._positive_int(spec.get("output_dim"), "Linear output_dim")
        return Linear(input_dim, output_dim), output_dim

    def build_sequential(
        self, spec: ArchitectureSpec, input_dim: int
    ) -> tuple[Sequential, int]:
        components: list[Component] = []
        dim = input_dim

        for layer in self._children(spec, "layers"):
            component, dim = self.build(layer, dim)
            components.append(component)

        return Sequential(components), dim

    def build_concat(
        self, spec: ArchitectureSpec, input_dim: int
    ) -> tuple[Concat, int]:
        branches: list[Component] = []
        output_dims: list[int] = []

        for path in self._children(spec, "paths"):
            branch, output_dim = self.build(path, input_dim)
            branches.append(branch)
            output_dims.append(output_dim)

        return Concat(branches, output_dims), sum(output_dims)

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _validate_mapping(value: object, name: str) -> ArchitectureSpec:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a YAML mapping")
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        return value

    @staticmethod
    def _children(spec: ArchitectureSpec, key: str) -> list[ArchitectureSpec]:
        value = spec.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"'{key}' must be a nonempty list")

        children: list[ArchitectureSpec] = []

        for index, child in enumerate(value):
            children.append(Builder._validate_mapping(child, f"'{key}[{index}]'"))

        return children
