from collections.abc import Mapping
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from coppercore.shared import FloatArray

from ..parameter import Parameter


def to_numpy_array(tensor: object) -> np.ndarray:
    if hasattr(tensor, "get"):
        return np.asarray(tensor.get())
    return np.array(tensor, copy=True)


def extract_state_dict(
    named_params: list[tuple[str, Parameter]],
) -> dict[str, FloatArray]:
    return {name: to_numpy_array(param.data) for name, param in named_params}


def validate_and_load_state_dict(
    named_params: list[tuple[str, Parameter]],
    state_dict: Mapping[str, object],
) -> None:
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"State dict must be a mapping, got {type(state_dict).__name__}"
        )

    expected_keys = {name for name, _ in named_params}
    provided_keys = set(state_dict.keys())

    missing_keys = expected_keys - provided_keys
    unexpected_keys = provided_keys - expected_keys

    if missing_keys or unexpected_keys:
        errors = []
        if missing_keys:
            errors.append(f"missing keys: {sorted(missing_keys)}")
        if unexpected_keys:
            errors.append(f"unexpected keys: {sorted(unexpected_keys)}")
        raise ValueError(f"State dict validation failed: {'; '.join(errors)}")

    validated_arrays: dict[str, np.ndarray] = {}

    for name, param in named_params:
        value = state_dict[name]
        if not isinstance(value, np.ndarray) and not isinstance(
            value, (list, tuple, int, float)
        ):
            raise TypeError(
                f"Parameter '{name}' value must be an array-like object, got {type(value).__name__}"
            )
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"Parameter '{name}' must have a numeric dtype, got {arr.dtype}"
            )
        if arr.shape != param.data.shape:
            raise ValueError(
                f"Shape mismatch for parameter '{name}': expected {param.data.shape}, got {arr.shape}"
            )
        validated_arrays[name] = arr

    for name, param in named_params:
        arr = validated_arrays[name]
        if hasattr(param.data, "__setitem__"):
            param.data[...] = arr
        else:
            param.data = np.array(arr, copy=True)


def save_weights_to_file(
    path: str | Path,
    state_dict: Mapping[str, FloatArray],
) -> None:
    dest_path = Path(path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=f".tmp_{dest_path.name}_",
        suffix=".npz",
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        np.savez(temp_path, **state_dict)
        temp_path.replace(dest_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_weights_from_file(path: str | Path) -> dict[str, FloatArray]:
    weight_path = Path(path).resolve()
    if not weight_path.exists() or not weight_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    try:
        with np.load(weight_path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except (zipfile.BadZipFile, ValueError, OSError) as err:
        raise ValueError(f"Corrupt or invalid weight file: {weight_path}") from err
