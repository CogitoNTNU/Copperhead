import re
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from .common import validate_mapping
from .types import Config, YamlMapping

_REFERENCE = re.compile(r"\$\{([^{}]+)\}")
type Lookup = Callable[[str], object]
type IncludeResolver = Callable[[YamlMapping], YamlMapping]


class Parser:
    def parse(
        self,
        path: str | Path,
        config: Config | None = None,
    ) -> YamlMapping:
        overrides = self._config_mapping(config, "Parser config")
        return self._load_file(Path(path), overrides, [])

    def _load_file(
        self,
        path: Path,
        overrides: Config,
        stack: list[Path],
    ) -> YamlMapping:
        path = path.resolve()

        if path in stack:
            cycle = " -> ".join(str(item) for item in [*stack, path])
            raise ValueError(f"Circular include detected: {cycle}")

        with path.open(encoding="utf-8") as file:
            document = validate_mapping(yaml.safe_load(file), f"YAML file {path}")

        defaults = self._config_mapping(document.get("config"), "File config")
        variables = self._resolve_config({**defaults, **overrides})
        body = {key: value for key, value in document.items() if key != "config"}

        def lookup(name: str) -> object:
            if name not in variables:
                raise ValueError(f"Missing configuration variable: {name!r}")
            return variables[name]

        def include(node: YamlMapping) -> YamlMapping:
            if set(node) - {"include", "config"}:
                raise ValueError(
                    "An include mapping may only contain 'include' and 'config'"
                )

            include_path = node["include"]
            if isinstance(include_path, str):
                include_path = self._resolve_string(include_path, lookup)

            if not isinstance(include_path, str) or not include_path.strip():
                raise TypeError("'include' must resolve to a nonblank string path")

            arguments = self._config_mapping(node.get("config"), "Include config")
            resolved_arguments = validate_mapping(
                self._walk(arguments, lookup),
                "Resolved include config",
            )

            return self._load_file(
                path.parent / include_path.strip(),
                resolved_arguments,
                [*stack, path],
            )

        return validate_mapping(
            self._walk(body, lookup, include=include),
            "Resolved architecture",
        )

    @staticmethod
    def _config_mapping(value: object, name: str) -> Config:
        if value is None:
            return {}
        return validate_mapping(value, name)

    def _resolve_config(self, values: Config) -> Config:
        resolved: dict[str, object] = {}
        resolving: list[str] = []

        def lookup(name: str) -> object:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                cycle = " -> ".join([*resolving, name])
                raise ValueError(f"Circular configuration reference: {cycle}")
            if name not in values:
                raise ValueError(f"Missing configuration variable: {name!r}")
            resolving.append(name)
            try:
                resolved[name] = self._walk(values[name], lookup)
                return resolved[name]
            finally:
                resolving.pop()

        for name in values:
            lookup(name)

        return resolved

    def _walk(
        self,
        value: object,
        lookup: Lookup,
        *,
        include: IncludeResolver | None = None,
        active: set[int] | None = None,
    ) -> object:
        if isinstance(value, str):
            return self._resolve_string(value, lookup)

        if not isinstance(value, (Mapping, list)):
            return value

        if active is None:
            active = set()

        identity = id(value)
        if identity in active:
            raise ValueError("Recursive YAML alias detected")

        active.add(identity)
        try:
            if isinstance(value, list):
                return [
                    self._walk(item, lookup, include=include, active=active)
                    for item in value
                ]

            mapping = validate_mapping(value, "YAML mapping")

            if include is not None and "include" in mapping:
                return include(mapping)

            return {
                key: self._walk(item, lookup, include=include, active=active)
                for key, item in mapping.items()
            }
        finally:
            active.remove(identity)

    @staticmethod
    def _resolve_string(value: str, lookup: Lookup) -> object:
        match = _REFERENCE.fullmatch(value)
        if match:
            return lookup(match.group(1))

        return _REFERENCE.sub(
            lambda match: str(lookup(match.group(1))),
            value,
        )
