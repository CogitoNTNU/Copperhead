from dataclasses import dataclass

from .common import positive_int, validate_mapping
from .types import YamlMapping


@dataclass(frozen=True)
class ValidatedArchitecture:
    name: str
    input_dim: int
    output_dim: int | None
    model: YamlMapping


class Validator:
    def validate_architecture(
        self,
        architecture: YamlMapping,
    ) -> ValidatedArchitecture:
        name = architecture.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Architecture name must be a nonblank string")

        input_dim = positive_int(
            architecture.get("input_dim"),
            "Architecture input_dim",
        )

        output_dim = (
            positive_int(architecture["output_dim"], "Architecture output_dim")
            if "output_dim" in architecture
            else None
        )

        model = validate_mapping(
            architecture.get("model"),
            "Architecture model",
        )

        if model.get("type") not in ("sequential", "concat"):
            raise ValueError("Architecture model must be 'sequential' or 'concat'")

        self._validate_model(model, expect_layer=True, path="model")

        return ValidatedArchitecture(name, input_dim, output_dim, model)

    def _validate_model(
        self,
        spec: YamlMapping,
        *,
        expect_layer: bool,
        path: str,
    ) -> bool:
        """Return whether the next component must be a layer."""
        component_type = spec.get("type")

        if not isinstance(component_type, str):
            raise TypeError(f"{path} requires a string 'type'")

        if component_type == "linear":
            if not expect_layer:
                raise ValueError(f"{path}: expected an activation, got a linear layer")

            positive_int(spec.get("output_dim"), f"{path}.output_dim")
            return False

        if component_type == "relu":
            if expect_layer:
                raise ValueError(f"{path}: expected a layer, got a ReLU activation")

            return True

        if component_type not in ("sequential", "concat"):
            raise TypeError(f"{path}: unsupported component type {component_type!r}")

        key = "layers" if component_type == "sequential" else "paths"
        children = spec.get(key)

        if not isinstance(children, list) or not children:
            raise ValueError(f"{path}.{key} must be a nonempty list")

        next_expectation = expect_layer
        branch_expectation: bool | None = None

        for index, child in enumerate(children):
            child_path = f"{path}.{key}[{index}]"
            child_spec = validate_mapping(child, child_path)

            if component_type == "sequential":
                next_expectation = self._validate_model(
                    child_spec,
                    expect_layer=next_expectation,
                    path=child_path,
                )
            else:
                result = self._validate_model(
                    child_spec,
                    expect_layer=expect_layer,
                    path=child_path,
                )

                if branch_expectation is None:
                    branch_expectation = result
                elif result != branch_expectation:
                    raise ValueError(
                        f"{child_path}: concat branches must all end "
                        "with a layer or all end with an activation"
                    )

                next_expectation = result

        return next_expectation
