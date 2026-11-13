import sys
from unittest.mock import MagicMock, patch

import coppercore.backend as backend_module
import coppercore.model_driven_nn.backend as mdnn_backend
import numpy as np
import pytest
from coppercore.backend import (
    get_array_module,
    get_backend,
    is_gpu_available,
    set_backend,
    to_device,
    to_numpy,
)
from coppercore.model_driven_nn.parameter import Parameter


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure backend is reset to cpu after each test."""
    yield
    backend_module._CURRENT_BACKEND = "cpu"
    backend_module._CUPY_MODULE = None
    backend_module._GPU_VALIDATED = False


def test_backend_reexport():
    """Test that model_driven_nn.backend re-exports all backend utilities."""
    assert mdnn_backend.get_backend is get_backend
    assert mdnn_backend.set_backend is set_backend
    assert mdnn_backend.is_gpu_available is is_gpu_available
    assert mdnn_backend.get_array_module is get_array_module
    assert mdnn_backend.to_device is to_device
    assert mdnn_backend.to_numpy is to_numpy


def test_get_and_set_backend_cpu():
    """Test get_backend and set_backend for cpu."""
    assert get_backend() == "cpu"
    set_backend("cpu")
    assert get_backend() == "cpu"


def test_set_backend_invalid():
    """Test set_backend with invalid backend name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend 'invalid'"):
        set_backend("invalid")


def test_set_backend_gpu_without_cuda():
    """Test set_backend('gpu') raises RuntimeError when GPU/CuPy is unavailable."""
    with (
        patch("coppercore.backend.is_gpu_available", return_value=False),
        pytest.raises(RuntimeError, match="CuPy / ROCm GPU runtime is not available"),
    ):
        set_backend("gpu")


def test_is_gpu_available_without_cupy():
    """is_gpu_available should return False when CuPy is not installed."""
    with patch.dict(sys.modules, {"cupy": None}):
        assert is_gpu_available() is False


def test_is_gpu_available_with_mocked_cupy():
    """Test is_gpu_available when CuPy and devices are mocked."""
    mock_cp = MagicMock()
    mock_cp.cuda.is_available.return_value = True
    mock_cp.cuda.runtime.getDeviceCount.return_value = 1

    with patch.dict("sys.modules", {"cupy": mock_cp}):
        assert is_gpu_available() is True

    mock_cp.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"cupy": mock_cp}):
        assert is_gpu_available() is False

    mock_cp.cuda.is_available.return_value = True
    mock_cp.cuda.runtime.getDeviceCount.side_effect = RuntimeError("No driver")
    with patch.dict("sys.modules", {"cupy": mock_cp}):
        assert is_gpu_available() is False


def test_get_array_module_cpu():
    """Test get_array_module for CPU numpy arrays and defaults."""
    arr = np.array([1.0, 2.0, 3.0])
    assert get_array_module() is np
    assert get_array_module(arr) is np
    assert get_array_module(arr, arr) is np
    assert get_array_module(None) is np


def test_get_array_module_mixed_devices_error():
    """Test get_array_module raises TypeError on mixed NumPy and CuPy inputs."""

    class MockCupyArray:
        __module__ = "cupy.ndarray"

    cp_arr = MockCupyArray()
    np_arr = np.array([1.0, 2.0])

    with pytest.raises(
        TypeError, match="Mixed device arrays passed to get_array_module"
    ):
        get_array_module(cp_arr, np_arr)

    with pytest.raises(
        TypeError, match="Mixed device arrays passed to get_array_module"
    ):
        get_array_module(np_arr, cp_arr)


def test_to_device_cpu():
    """Test to_device with cpu device and dtype conversion."""
    data = [1.0, 2.0, 3.0]
    out = to_device(data, device="cpu")
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))
    assert out.dtype == np.float64

    # Test with dtype
    out_f32 = to_device(out, device="cpu", dtype=np.float32)
    assert out_f32.dtype == np.float32

    # Default device should be cpu for numpy arrays
    out_default = to_device(out)
    assert isinstance(out_default, np.ndarray)


def test_to_device_invalid_device():
    """Test to_device raises ValueError on invalid device name."""
    with pytest.raises(ValueError, match="Invalid device 'tpu'"):
        to_device(np.array([1.0]), device="tpu")


def test_to_device_gpu_unavailable():
    """Test to_device raises RuntimeError when gpu requested without CuPy/GPU."""
    with (
        patch("coppercore.backend.is_gpu_available", return_value=False),
        pytest.raises(RuntimeError, match="CuPy / ROCm GPU support is not available"),
    ):
        to_device(np.array([1.0]), device="gpu")


def test_to_numpy():
    """Test to_numpy function."""
    data = [4.0, 5.0, 6.0]
    arr = to_numpy(data)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.array([4.0, 5.0, 6.0]))

    arr_f32 = to_numpy(arr, dtype=np.float32)
    assert arr_f32.dtype == np.float32


def test_parameter_cpu():
    """Test Parameter on CPU with gradient initialization, zero_grad, and to()."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    param = Parameter.from_data(data)

    assert isinstance(param.data, np.ndarray)
    assert isinstance(param.grad, np.ndarray)
    assert param.grad.shape == data.shape
    assert param.grad.dtype == data.dtype
    assert np.all(param.grad == 0.0)

    param.grad += 5.0
    assert np.all(param.grad == 5.0)
    param.zero_grad()
    assert np.all(param.grad == 0.0)

    # Test .to() with dtype conversion returns self and preserves identity
    res = param.to(device="cpu", dtype=np.float32)
    assert res is param
    assert param.data.dtype == np.float32
    assert param.grad.dtype == np.float32


def test_mock_cupy_gpu_flow():
    """Test GPU backend flows with a mocked CuPy module."""
    mock_cp = MagicMock()
    mock_cp.cuda.is_available.return_value = True
    mock_cp.cuda.runtime.getDeviceCount.return_value = 1

    # Mock cupy ndarray
    class MockCupyArray:
        __module__ = "cupy.ndarray"

        def __init__(self, data, dtype=None):
            self.data = np.array(data, dtype=dtype)
            self.dtype = self.data.dtype
            self.shape = self.data.shape

        def fill(self, val):
            self.data.fill(val)

        def astype(self, dtype, copy=False):
            return MockCupyArray(self.data.astype(dtype, copy=copy))

    mock_cp.ndarray = MockCupyArray
    mock_cp.asarray.side_effect = lambda a, **kwargs: (
        a
        if isinstance(a, MockCupyArray)
        else MockCupyArray(a, dtype=kwargs.get("dtype"))
    )
    mock_cp.asnumpy.side_effect = lambda a: (
        np.asarray(a.data) if isinstance(a, MockCupyArray) else np.asarray(a)
    )
    mock_cp.zeros_like.side_effect = lambda a: MockCupyArray(np.zeros_like(a.data))

    with patch.dict("sys.modules", {"cupy": mock_cp}):
        # Backend switching
        set_backend("gpu")
        assert get_backend() == "gpu"

        # Array module dispatch
        assert get_array_module() is mock_cp
        cp_arr = mock_cp.asarray([1.0, 2.0])
        assert get_array_module(cp_arr) is mock_cp
        assert get_array_module(np.array([1.0])) is np

        # to_device transfers
        gpu_arr = to_device(np.array([1.0, 2.0]), device="gpu")
        assert isinstance(gpu_arr, MockCupyArray)

        # to_device with device=None retains GPU residency
        gpu_arr_f32 = to_device(gpu_arr, dtype=np.float32)
        assert isinstance(gpu_arr_f32, MockCupyArray)
        assert gpu_arr_f32.dtype == np.float32
        assert mock_cp.cuda.runtime.getDeviceCount.call_count == 1

        cpu_arr = to_device(gpu_arr, device="cpu")
        assert isinstance(cpu_arr, np.ndarray)
        assert np.array_equal(cpu_arr, np.array([1.0, 2.0]))

        # to_numpy from GPU array
        np_arr = to_numpy(gpu_arr)
        assert isinstance(np_arr, np.ndarray)
        assert np.array_equal(np_arr, np.array([1.0, 2.0]))

        # Parameter on GPU
        gpu_param = Parameter.from_data(gpu_arr)
        assert isinstance(gpu_param.data, MockCupyArray)
        assert isinstance(gpu_param.grad, MockCupyArray)

        # Device retention on Parameter.to(dtype=...) while global backend is "cpu"
        set_backend("cpu")
        assert get_backend() == "cpu"
        ret = gpu_param.to(dtype=np.float32)
        assert ret is gpu_param
        assert isinstance(gpu_param.data, MockCupyArray)
        assert isinstance(gpu_param.grad, MockCupyArray)
        assert gpu_param.data.dtype == np.float32
        assert gpu_param.grad.dtype == np.float32

        # Move parameter to cpu
        gpu_param.to(device="cpu")
        assert isinstance(gpu_param.data, np.ndarray)
        assert isinstance(gpu_param.grad, np.ndarray)


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy/GPU is not available")
def test_real_gpu_device_and_parameter():
    """Verify device transfers and Parameter.to on actual GPU hardware if present."""
    import cupy as cp

    # Backend switching
    set_backend("gpu")
    assert get_backend() == "gpu"
    assert get_array_module() is cp

    x_cpu = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x_gpu = to_device(x_cpu, device="gpu")
    assert isinstance(x_gpu, cp.ndarray)
    assert get_array_module(x_gpu) is cp

    # Parameter on GPU
    param = Parameter.from_data(x_gpu)
    assert isinstance(param.data, cp.ndarray)
    assert isinstance(param.grad, cp.ndarray)

    # Change dtype without altering GPU residency when backend is cpu
    set_backend("cpu")
    param.to(dtype=np.float32)
    assert isinstance(param.data, cp.ndarray)
    assert param.data.dtype == np.float32
    assert param.grad.dtype == np.float32

    # Move back to CPU
    param.to(device="cpu")
    assert isinstance(param.data, np.ndarray)
    assert isinstance(param.grad, np.ndarray)
