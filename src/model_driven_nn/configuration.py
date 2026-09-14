"""YAML configuration and component-file composition."""

import re
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from .common import validate_mapping
from .types import ArchitectureSpec

_REFERENCE = re.compile(r"\$\{([^{}]+)\}")


class ConfigurationLoader:
    """Resolve variables and includes before handing a mapping to the builder."""

    def load(
        self, path: str | Path, config: Mapping[str, object] | None = None
    ) -> ArchitectureSpec:
        inherited = dict(validate_mapping(config or {}, "Loader config"))
        return self._load_file(Path(path), inherited, [])

    def _load_file(
        self,
        path: Path,
        inherited: Mapping[str, object],
        stack: list[Path],
        overrides: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        path = path.resolve()
        if path in stack:
            cycle = " -> ".join(str(item) for item in [*stack, path])
            raise ValueError(f"Circular include detected: {cycle}")
        if not path.is_file():
            kind = "Included" if stack else "Architecture"
            raise FileNotFoundError(f"{kind} file not found: {path}")

        with path.open(encoding="utf-8") as file:
            document = validate_mapping(yaml.safe_load(file), f"YAML file {path}")

        defaults = self._mapping(document.get("config", {}), f"config in {path}")
        merged = {**defaults, **inherited, **dict(overrides or {})}
        variables = self._resolve_config(merged)
        body = {key: value for key, value in document.items() if key != "config"}
        return self._resolve_node(body, variables, path.parent, [*stack, path], set())

    @staticmethod
    def _mapping(value: object, name: str) -> dict[str, object]:
        if value is None:
            return {}
        return dict(validate_mapping(value, name))

    def _resolve_config(self, values: Mapping[str, object]) -> dict[str, object]:
        resolved: dict[str, object] = {}
        resolving: list[str] = []

        def resolve_name(name: str) -> object:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                cycle = " -> ".join([*resolving, name])
                raise ValueError(f"Circular configuration reference: {cycle}")
            if name not in values:
                raise ValueError(f"Missing configuration variable: {name!r}")
            resolving.append(name)
            result = self._resolve_value(values[name], resolve_name, set())
            resolving.pop()
            resolved[name] = result
            return result

        return {name: resolve_name(name) for name in values}

    def _resolve_value(
        self,
        value: object,
        resolve_name: Callable[[str], object],
        active: set[int],
    ) -> object:
        if isinstance(value, str):
            return self._resolve_string(value, resolve_name)
        if isinstance(value, (Mapping, list)):
            identity = id(value)
            if identity in active:
                raise ValueError("Recursive YAML alias detected in configuration")
            active.add(identity)
            try:
                if isinstance(value, Mapping):
                    return {
                        key: self._resolve_value(item, resolve_name, active)
                        for key, item in value.items()
                    }
                return [
                    self._resolve_value(item, resolve_name, active) for item in value
                ]
            finally:
                active.remove(identity)
        return value

    @staticmethod
    def _resolve_string(value: str, resolve_name: Callable[[str], object]) -> object:
        full = re.fullmatch(_REFERENCE, value)
        if full:
            return resolve_name(full.group(1))
        return _REFERENCE.sub(lambda match: str(resolve_name(match.group(1))), value)

    def _resolve_node(
        self,
        node: object,
        config: Mapping[str, object],
        base_dir: Path,
        stack: list[Path],
        active: set[int],
    ) -> object:
        if isinstance(node, (Mapping, list)):
            identity = id(node)
            if identity in active:
                raise ValueError("Recursive YAML alias detected in architecture")
            active.add(identity)
            try:
                if isinstance(node, list):
                    return [
                        self._resolve_node(item, config, base_dir, stack, active)
                        for item in node
                    ]
                return self._resolve_mapping(node, config, base_dir, stack, active)
            finally:
                active.remove(identity)
        if isinstance(node, str):
            return self._resolve_string(
                node, lambda name: self._config_value(config, name)
            )
        return node

    def _resolve_mapping(
        self,
        node: Mapping[str, object],
        config: Mapping[str, object],
        base_dir: Path,
        stack: list[Path],
        active: set[int],
    ) -> dict[str, object]:
        if "include" in node:
            extras = set(node) - {"include", "config"}
            if extras:
                raise ValueError(
                    "An include mapping may only contain 'include' and 'config'"
                )
            include_value = node["include"]
            if isinstance(include_value, str):
                include_value = self._resolve_string(
                    include_value, lambda name: self._config_value(config, name)
                )
            if not isinstance(include_value, str) or not include_value.strip():
                raise TypeError("'include' must resolve to a nonblank string path")
            raw_overrides = self._mapping(node.get("config", {}), "include config")
            overrides = self._resolve_value(
                raw_overrides, lambda name: self._config_value(config, name), set()
            )
            overrides = validate_mapping(overrides, "resolved include config")
            return self._load_file(
                base_dir / include_value.strip(), config, stack, overrides
            )
        return {
            key: self._resolve_node(value, config, base_dir, stack, active)
            for key, value in node.items()
        }

    @staticmethod
    def _config_value(config: Mapping[str, object], name: str) -> object:
        if name not in config:
            raise ValueError(f"Missing configuration variable: {name!r}")
        return config[name]
