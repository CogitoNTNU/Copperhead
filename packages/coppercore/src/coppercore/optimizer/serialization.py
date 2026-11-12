import json
from pathlib import Path
from typing import Any

import numpy as np

from coppercore.model_driven_nn.serialization.weights import to_numpy_array


def save_optimizer_state_to_file(
    path: str | Path,
    state_dict: dict[str, Any],
) -> None:
    arrays = {}

    def encode(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]

        if isinstance(value, np.ndarray) or hasattr(value, "get"):
            key = f"array_{len(arrays)}"
            arrays[key] = to_numpy_array(value)
            return {"__array__": key}

        return value

    metadata = json.dumps(encode(state_dict))

    np.savez(
        path,
        __metadata__=metadata,
        **arrays,
    )


def load_optimizer_state_from_file(
    path: str | Path,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["__metadata__"].item())

        def decode(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    return archive[value["__array__"]].copy()

                return {k: decode(v) for k, v in value.items()}

            if isinstance(value, list):
                return [decode(v) for v in value]

            return value

        return decode(metadata)
