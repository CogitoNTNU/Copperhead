"""Centralized array backend supporting NumPy (CPU) and CuPy (GPU with ROCm/CUDA)."""

import importlib
import types
from typing import Any

import numpy as np

_CURRENT_BACKEND: str = "cpu"
_SUPPORTED_BACKENDS: tuple[str, ...] = ("cpu", "gpu")
_CUPY_MODULE: types.ModuleType | None = None
_GPU_VALIDATED: bool = False

_GPU_INSTALL_HELP: str = (
    "Please install CuPy matching your GPU platform, e.g.:\n"
    "  - AMD ROCm 7.x: pip install cupy-rocm-7-0\n"
    "  - AMD ROCm 5.x: pip install cupy-rocm-5-0\n"
    "  - NVIDIA CUDA 12.x: pip install cupy-cuda12x\n"
    "  - From source for ROCm: set CUPY_INSTALL_USE_HIP=1, ROCM_HOME, and "
    "HCC_AMDGPU_TARGET before installing cupy\n"
    "See https://docs.cupy.dev/en/stable/install.html for installation instructions."
)


def is_gpu_available() -> bool:
    """Check if CuPy is installed and a compatible GPU device is accessible.

    Returns:
        bool: True if CuPy and a GPU device are available, False otherwise.
    """
    try:
        cp = importlib.import_module("cupy")
        if hasattr(cp, "cuda") and cp.cuda.is_available():
            try:
                return bool(cp.cuda.runtime.getDeviceCount() > 0)
            except (RuntimeError, AttributeError):
                return False
        return False
    except (ImportError, AttributeError, RuntimeError):
        return False


def _get_cupy_module() -> types.ModuleType | None:
    """Get the cached CuPy module or import it if available."""
    global _CUPY_MODULE
    if _CUPY_MODULE is not None:
        return _CUPY_MODULE
    try:
        cp = importlib.import_module("cupy")
        _CUPY_MODULE = cp
        return _CUPY_MODULE
    except (ImportError, AttributeError):
        return None


def get_backend() -> str:
    """Get the currently configured backend name ('cpu' or 'gpu').

    Returns:
        str: Current backend name.
    """
    return _CURRENT_BACKEND


def set_backend(name: str) -> None:
    """Set the active backend ('cpu' or 'gpu').

    Args:
        name: Name of the backend ('cpu' or 'gpu').

    Raises:
        ValueError: If backend name is not 'cpu' or 'gpu'.
        RuntimeError: If 'gpu' is requested but CuPy/GPU is not available.
    """
    global _CURRENT_BACKEND, _CUPY_MODULE, _GPU_VALIDATED
    normalized = name.strip().lower()
    if normalized not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{name}'. Supported backends are: {list(_SUPPORTED_BACKENDS)}"
        )

    if normalized == "gpu":
        if not is_gpu_available():
            raise RuntimeError(
                f"GPU backend requested ('gpu'), but CuPy / ROCm GPU runtime is not available.\n"
                f"{_GPU_INSTALL_HELP}"
            )
        import cupy as cp

        _CUPY_MODULE = cp
        _GPU_VALIDATED = True
    else:
        _CUPY_MODULE = None
        _GPU_VALIDATED = False

    _CURRENT_BACKEND = normalized


def _is_cupy_array(array: Any) -> bool:
    """Check if an object is a CuPy ndarray without requiring CuPy to be top-level imported."""
    if array is None:
        return False
    type_module = getattr(type(array), "__module__", "")
    return type_module.startswith("cupy")


def get_array_module(*arrays: Any) -> types.ModuleType:
    """Return the appropriate array module (numpy or cupy) for the given arrays or backend.

    If arrays are provided, inspects their device residency:
    - If all valid arrays are NumPy ndarrays, returns `numpy`.
    - If all valid arrays are CuPy ndarrays, returns `cupy`.
    - If a mix of NumPy and CuPy arrays is provided, raises a TypeError.
    - If no arrays are provided (or all are None), returns the module corresponding
      to the current backend (`set_backend`).

    Args:
        *arrays: Optional array objects or tensors.

    Returns:
        ModuleType: numpy or cupy module.

    Raises:
        TypeError: If mixed NumPy and CuPy arrays are passed.
    """
    valid_arrays = [a for a in arrays if a is not None]

    if valid_arrays:
        has_cupy = False
        has_numpy = False
        for a in valid_arrays:
            if _is_cupy_array(a):
                has_cupy = True
            elif isinstance(a, np.ndarray):
                has_numpy = True

        if has_cupy and has_numpy:
            raise TypeError(
                "Mixed device arrays passed to get_array_module: received both NumPy (CPU) "
                "and CuPy (GPU) arrays. Explicitly transfer all inputs to the same device "
                "using to_device()."
            )

        if has_cupy:
            cp = _get_cupy_module()
            if cp is not None:
                return cp
            import cupy as cp_imported

            return cp_imported

        if has_numpy:
            return np

        # For non-ndarray objects (like lists or scalars), check via cupy if available
        cp = _get_cupy_module()
        if cp is not None:
            try:
                return cp.get_array_module(*valid_arrays)
            except AttributeError:
                pass
        return np

    if _CURRENT_BACKEND == "gpu":
        cp = _get_cupy_module()
        if cp is not None:
            return cp

    return np


def to_device(
    array: Any,
    device: str | None = None,
    dtype: Any = None,
) -> Any:
    """Move an array or array-like object to the specified device and optional dtype.

    If `device` is None, preserves the array's current device:
    - If `array` is a CuPy array, keeps it on 'gpu'.
    - If `array` is a NumPy array, keeps it on 'cpu'.
    - Otherwise, falls back to the current backend (`get_backend()`).

    Args:
        array: Input array or array-like object (numpy ndarray, cupy ndarray, list, scalar).
        device: Target device ('cpu' or 'gpu'). If None, preserves array's device or active backend.
        dtype: Optional target data type (e.g. np.float32, np.float64, 'float32').

    Returns:
        Array on the target device with the specified dtype.

    Raises:
        ValueError: If device is not 'cpu' or 'gpu'.
        RuntimeError: If target device is 'gpu' but GPU support is unavailable.
    """
    if device is None:
        if _is_cupy_array(array):
            target_device = "gpu"
        elif isinstance(array, np.ndarray):
            target_device = "cpu"
        else:
            target_device = _CURRENT_BACKEND
    else:
        target_device = device.strip().lower()

    if target_device not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Invalid device '{device}'. Expected one of {list(_SUPPORTED_BACKENDS)}."
        )

    if target_device == "cpu":
        if _is_cupy_array(array):
            import cupy as cp

            out = cp.asnumpy(array)
        elif isinstance(array, np.ndarray):
            out = array
        else:
            out = np.asarray(array)

        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    elif target_device == "gpu":
        global _CUPY_MODULE, _GPU_VALIDATED
        if not _GPU_VALIDATED or _CUPY_MODULE is None:
            if not is_gpu_available():
                raise RuntimeError(
                    f"Cannot move array to 'gpu': CuPy / ROCm GPU support is not available.\n"
                    f"{_GPU_INSTALL_HELP}"
                )
            _CUPY_MODULE = _get_cupy_module()
            _GPU_VALIDATED = _CUPY_MODULE is not None
        cp = _CUPY_MODULE
        if cp is None:
            raise RuntimeError(
                f"Cannot move array to 'gpu': CuPy / ROCm GPU support is not available.\n"
                f"{_GPU_INSTALL_HELP}"
            )

        if _is_cupy_array(array):
            out = array
        else:
            out = cp.asarray(array)

        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out


def to_numpy(array: Any, dtype: Any = None) -> np.ndarray:
    """Convert any array (numpy, cupy, list, scalar) to a standard CPU NumPy ndarray.

    Args:
        array: Input array or array-like.
        dtype: Optional target data type.

    Returns:
        np.ndarray: CPU NumPy ndarray.
    """
    return to_device(array, device="cpu", dtype=dtype)
