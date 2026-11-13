import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml


def save_architecture_to_file(
    path: str | Path,
    config: Mapping[str, object],
) -> None:
    dest_path = Path(path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=f".tmp_{dest_path.name}_",
        suffix=".yaml",
        delete=False,
        mode="w",
        encoding="utf-8",
    )
    temp_path = Path(temp_file.name)

    try:
        yaml.safe_dump(config, temp_file, sort_keys=False)
        temp_file.close()
        temp_path.replace(dest_path)
    except Exception:
        temp_file.close()
        if temp_path.exists():
            temp_path.unlink()
        raise
