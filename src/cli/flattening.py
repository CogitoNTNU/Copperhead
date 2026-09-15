from pathlib import Path

import yaml

from model_driven_nn.parser import Parser
from model_driven_nn.types import Config
from model_driven_nn.validator import Validator


def flatten_yaml(
    source: str | Path,
    destination: str | Path,
    config: Config | None = None,
) -> None:
    architecture = Parser().parse(source, config=config)
    Validator().validate_architecture(architecture)

    with Path(destination).open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            dict(architecture),
            file,
            sort_keys=False,
            allow_unicode=True,
        )
