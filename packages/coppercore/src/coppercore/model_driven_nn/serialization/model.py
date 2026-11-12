import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..network import Network

FORMAT_VERSION = 1


def save_model_bundle(network: "Network", dir_path: str | Path) -> None:
    dest_dir = Path(dir_path).resolve()
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        tempfile.mkdtemp(dir=dest_dir.parent, prefix=f".tmp_{dest_dir.name}_")
    )

    try:
        arch_filename = "model.yaml"
        weights_filename = "weights.npz"
        manifest_filename = "manifest.json"

        network.save_architecture(temp_dir / arch_filename)
        network.save_weights(temp_dir / weights_filename)

        manifest = {
            "format_version": FORMAT_VERSION,
            "architecture": arch_filename,
            "weights": weights_filename,
        }

        manifest_path = temp_dir / manifest_filename
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

        if dest_dir.exists():
            backup_dir = Path(
                tempfile.mkdtemp(
                    dir=dest_dir.parent, prefix=f".backup_{dest_dir.name}_"
                )
            )
            backup_dir.rmdir()
            dest_dir.replace(backup_dir)
            try:
                temp_dir.replace(dest_dir)
                shutil.rmtree(backup_dir, ignore_errors=True)
            except Exception:
                if backup_dir.exists() and not dest_dir.exists():
                    backup_dir.replace(dest_dir)
                raise
        else:
            temp_dir.replace(dest_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_model_bundle(
    dir_path: str | Path,
    device: str | None = None,
    dtype: Any = None,
) -> "Network":
    from ..construction.loader import Loader
    from .weights import load_weights_from_file

    model_dir = Path(dir_path).resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except Exception as err:
        raise ValueError(f"Invalid manifest JSON in {manifest_path}") from err

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")

    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported serialization format version: {version}. "
            f"Expected format version {FORMAT_VERSION}."
        )

    arch_file = manifest.get("architecture")
    if not isinstance(arch_file, str) or not arch_file.strip():
        raise ValueError("Manifest missing or invalid 'architecture' field")

    weights_file = manifest.get("weights")
    if not isinstance(weights_file, str) or not weights_file.strip():
        raise ValueError("Manifest missing or invalid 'weights' field")

    arch_path = model_dir / arch_file
    if not arch_path.exists():
        raise FileNotFoundError(
            f"Architecture file specified in manifest not found: {arch_path}"
        )

    weights_path = model_dir / weights_file
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weight file specified in manifest not found: {weights_path}"
        )

    weights = load_weights_from_file(weights_path)
    target_dtype = dtype
    if target_dtype is None and weights:
        target_dtype = next(iter(weights.values())).dtype

    network = Loader().load(arch_path)
    if device is not None or target_dtype is not None:
        network.to(device=device, dtype=target_dtype)
    network.load_state_dict(weights)
    return network
