from unittest.mock import MagicMock, patch

import coppercore.backend as backend_module
import numpy as np
import pytest
from coppercore.backend import (
    is_gpu_available,
    set_backend,
    to_device,
    to_numpy,
)
from coppercore.model_driven_nn import Network
from coppercore.model_driven_nn.components import (
    AvgPool2D,
    Concat,
    Conv2D,
    Flatten,
    Linear,
    MaxPool2D,
    ReLU,
    Sequential,
)
from coppercore.model_driven_nn.components.im2col import col2im, im2col
from coppercore.model_driven_nn.parameter import Parameter
from coppercore.optimizer import ADAM, SGD
from coppercore.training.dqn_train import dqn_mse
from coppercore.training.mnist_train import accuracy, softmax_cross_entropy_loss


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure backend is reset to cpu after each test."""
    yield
    backend_module._CURRENT_BACKEND = "cpu"
    backend_module._CUPY_MODULE = None
    backend_module._GPU_VALIDATED = False


def test_network_and_components_to_dtype():
    """Verify recursive .to(dtype=...) propagation across complex network hierarchies."""
    net = Network(
        name="test_cnn",
        model=Sequential(
            [
                Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1),
                ReLU(),
                MaxPool2D(kernel_size=2),
                Concat(
                    [
                        Sequential([Conv2D(in_channels=4, out_channels=4, kernel_size=3, padding=1), ReLU()]),
                        Conv2D(in_channels=4, out_channels=4, kernel_size=1),
                    ]
                ),
                AvgPool2D(kernel_size=2),
                Flatten(),
                Linear(input_dim=4 * 2 * 4, output_dim=2),
            ]
        ),
        input_dim=64,
        output_dim=2,
    )

    # Initial float64
    for p in net.params():
        assert p.data.dtype == np.float64
        assert p.grad.dtype == np.float64

    # Convert network to float32
    ret = net.to(dtype=np.float32)
    assert ret is net

    for p in net.params():
        assert p.data.dtype == np.float32
        assert p.grad.dtype == np.float32

    # Forward and backward in float32
    x = np.random.randn(2, 64).astype(np.float32)
    out = net.forward(x, cache=True)
    assert out.dtype == np.float32
    assert out.shape == (2, 2)

    d_out = np.ones_like(out, dtype=np.float32)
    d_x = net.backward(d_out)
    assert d_x.dtype == np.float32
    assert d_x.shape == x.shape

    for p in net.params():
        assert p.grad.dtype == np.float32


def test_linear_and_relu_to_device():
    """Test Linear and ReLU forward/backward and .to(device) semantics."""
    fc = Linear(4, 2)
    relu = ReLU()

    fc.to(dtype=np.float32)
    assert fc.W.data.dtype == np.float32
    assert fc.b.data.dtype == np.float32

    x = np.array([[1.0, -2.0, 3.0, -4.0], [-1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
    out_fc = fc.forward(x, cache=True)
    out_relu = relu.forward(out_fc, cache=True)

    assert out_relu.dtype == np.float32
    assert np.all(out_relu >= 0.0)

    d_y = np.ones_like(out_relu, dtype=np.float32)
    d_relu = relu.backward(d_y)
    d_fc = fc.backward(d_relu)

    assert d_fc.dtype == np.float32
    assert d_fc.shape == x.shape
    assert fc.W.grad.dtype == np.float32
    assert fc.b.grad.dtype == np.float32


def test_concat_forward_backward():
    """Test Concat layer with multiple branches."""
    branch1 = Linear(4, 3)
    branch2 = Sequential([Linear(4, 2), ReLU()])
    concat = Concat([branch1, branch2])

    concat.to(dtype=np.float32)
    assert len(concat.params()) == 4
    for p in concat.params():
        assert p.data.dtype == np.float32

    x = np.random.randn(2, 4).astype(np.float32)
    out = concat.forward(x, cache=True)
    assert out.shape == (2, 5)
    assert out.dtype == np.float32

    d_out = np.ones_like(out, dtype=np.float32)
    d_x = concat.backward(d_out)
    assert d_x.shape == x.shape
    assert d_x.dtype == np.float32


def test_spatial_ops_cpu_float32():
    """Test Conv2D, MaxPool2D, AvgPool2D with im2col/col2im in float32."""
    conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    maxpool = MaxPool2D(kernel_size=2, stride=2)
    avgpool = AvgPool2D(kernel_size=2, stride=2)

    conv.to(dtype=np.float32)
    maxpool.to(dtype=np.float32)
    avgpool.to(dtype=np.float32)

    x = np.random.randn(2, 2, 4, 4).astype(np.float32)

    # Conv2D
    out_conv = conv.forward(x, cache=True)
    assert out_conv.shape == (2, 3, 4, 4)
    assert out_conv.dtype == np.float32

    d_conv = conv.backward(np.ones_like(out_conv))
    assert d_conv.shape == x.shape
    assert d_conv.dtype == np.float32

    # MaxPool2D
    out_max = maxpool.forward(x, cache=True)
    assert out_max.shape == (2, 2, 2, 2)
    assert out_max.dtype == np.float32

    d_max = maxpool.backward(np.ones_like(out_max))
    assert d_max.shape == x.shape
    assert d_max.dtype == np.float32

    # AvgPool2D
    out_avg = avgpool.forward(x, cache=True)
    assert out_avg.shape == (2, 2, 2, 2)
    assert out_avg.dtype == np.float32

    d_avg = avgpool.backward(np.ones_like(out_avg))
    assert d_avg.shape == x.shape
    assert d_avg.dtype == np.float32


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy/GPU is not available")
def test_real_gpu_layers_numerical_parity():
    """Verify numerical parity of forward and backward passes between CPU and GPU."""
    import cupy as cp

    rng = np.random.default_rng(42)

    # 1. Linear
    fc_cpu = Linear(8, 4)
    fc_gpu = Linear(8, 4)
    fc_gpu.W.data = to_device(fc_cpu.W.data.copy(), device="gpu")
    fc_gpu.W.grad = to_device(fc_cpu.W.grad.copy(), device="gpu")
    fc_gpu.b.data = to_device(fc_cpu.b.data.copy(), device="gpu")
    fc_gpu.b.grad = to_device(fc_cpu.b.grad.copy(), device="gpu")

    x_cpu = rng.standard_normal((3, 8))
    x_gpu = to_device(x_cpu, device="gpu")

    out_cpu = fc_cpu.forward(x_cpu, cache=True)
    out_gpu = fc_gpu.forward(x_gpu, cache=True)
    assert isinstance(out_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(out_gpu), out_cpu, rtol=1e-5, atol=1e-5)

    d_y_cpu = rng.standard_normal(out_cpu.shape)
    d_y_gpu = to_device(d_y_cpu, device="gpu")

    dx_cpu = fc_cpu.backward(d_y_cpu)
    dx_gpu = fc_gpu.backward(d_y_gpu)
    assert isinstance(dx_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(dx_gpu), dx_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(fc_gpu.W.grad), fc_cpu.W.grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(fc_gpu.b.grad), fc_cpu.b.grad, rtol=1e-5, atol=1e-5)

    # 2. ReLU
    relu_cpu = ReLU()
    relu_gpu = ReLU()

    x_relu_cpu = rng.standard_normal((3, 8))
    x_relu_gpu = to_device(x_relu_cpu, device="gpu")

    out_relu_cpu = relu_cpu.forward(x_relu_cpu, cache=True)
    out_relu_gpu = relu_gpu.forward(x_relu_gpu, cache=True)
    assert isinstance(out_relu_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(out_relu_gpu), out_relu_cpu, rtol=1e-5, atol=1e-5)

    dx_relu_cpu = relu_cpu.backward(d_y_cpu)
    dx_relu_gpu = relu_gpu.backward(d_y_gpu)
    assert isinstance(dx_relu_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(dx_relu_gpu), dx_relu_cpu, rtol=1e-5, atol=1e-5)

    # 3. Conv2D
    conv_cpu = Conv2D(in_channels=2, out_channels=3, kernel_size=3, padding=1, stride=1, seed=123)
    conv_gpu = Conv2D(in_channels=2, out_channels=3, kernel_size=3, padding=1, stride=1, seed=123)
    conv_gpu.to(device="gpu")

    x_conv_cpu = rng.standard_normal((2, 2, 6, 6))
    x_conv_gpu = to_device(x_conv_cpu, device="gpu")

    out_conv_cpu = conv_cpu.forward(x_conv_cpu, cache=True)
    out_conv_gpu = conv_gpu.forward(x_conv_gpu, cache=True)
    assert isinstance(out_conv_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(out_conv_gpu), out_conv_cpu, rtol=1e-5, atol=1e-5)

    d_out_conv_cpu = rng.standard_normal(out_conv_cpu.shape)
    d_out_conv_gpu = to_device(d_out_conv_cpu, device="gpu")

    dx_conv_cpu = conv_cpu.backward(d_out_conv_cpu)
    dx_conv_gpu = conv_gpu.backward(d_out_conv_gpu)
    assert isinstance(dx_conv_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(dx_conv_gpu), dx_conv_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(conv_gpu.W.grad), conv_cpu.W.grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(conv_gpu.b.grad), conv_cpu.b.grad, rtol=1e-5, atol=1e-5)

    # 4. MaxPool2D
    max_cpu = MaxPool2D(kernel_size=2, stride=2, padding=0)
    max_gpu = MaxPool2D(kernel_size=2, stride=2, padding=0)
    max_gpu.to(device="gpu")

    out_max_cpu = max_cpu.forward(x_conv_cpu, cache=True)
    out_max_gpu = max_gpu.forward(x_conv_gpu, cache=True)
    assert isinstance(out_max_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(out_max_gpu), out_max_cpu, rtol=1e-5, atol=1e-5)

    d_out_max_cpu = rng.standard_normal(out_max_cpu.shape)
    d_out_max_gpu = to_device(d_out_max_cpu, device="gpu")

    dx_max_cpu = max_cpu.backward(d_out_max_cpu)
    dx_max_gpu = max_gpu.backward(d_out_max_gpu)
    assert isinstance(dx_max_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(dx_max_gpu), dx_max_cpu, rtol=1e-5, atol=1e-5)

    # 5. AvgPool2D
    avg_cpu = AvgPool2D(kernel_size=2, stride=2, padding=0)
    avg_gpu = AvgPool2D(kernel_size=2, stride=2, padding=0)
    avg_gpu.to(device="gpu")

    out_avg_cpu = avg_cpu.forward(x_conv_cpu, cache=True)
    out_avg_gpu = avg_gpu.forward(x_conv_gpu, cache=True)
    assert isinstance(out_avg_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(out_avg_gpu), out_avg_cpu, rtol=1e-5, atol=1e-5)

    d_out_avg_cpu = rng.standard_normal(out_avg_cpu.shape)
    d_out_avg_gpu = to_device(d_out_avg_cpu, device="gpu")

    dx_avg_cpu = avg_cpu.backward(d_out_avg_cpu)
    dx_avg_gpu = avg_gpu.backward(d_out_avg_gpu)
    assert isinstance(dx_avg_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(dx_avg_gpu), dx_avg_cpu, rtol=1e-5, atol=1e-5)

    # 6. im2col and col2im
    x_pad_cpu = rng.standard_normal((2, 3, 6, 6))
    x_pad_gpu = to_device(x_pad_cpu, device="gpu")
    cols_cpu = im2col(x_pad_cpu, 3, 3, 1, 1, 4, 4)
    cols_gpu = im2col(x_pad_gpu, 3, 3, 1, 1, 4, 4)
    assert isinstance(cols_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(cols_gpu), cols_cpu, rtol=1e-5, atol=1e-5)

    rec_cpu = col2im(cols_cpu, x_pad_cpu.shape, 3, 3, 1, 1, 4, 4)
    rec_gpu = col2im(cols_gpu, x_pad_gpu.shape, 3, 3, 1, 1, 4, 4)
    assert isinstance(rec_gpu, cp.ndarray)
    np.testing.assert_allclose(to_numpy(rec_gpu), rec_cpu, rtol=1e-5, atol=1e-5)


def test_mocked_cupy_network_to_gpu():
    """Verify Network.to('gpu') and forward/backward with mocked CuPy."""
    mock_cp = MagicMock()
    mock_cp.cuda.is_available.return_value = True
    mock_cp.cuda.runtime.getDeviceCount.return_value = 1

    class MockCupyArray:
        __module__ = "cupy.ndarray"

        def __init__(self, data, dtype=None):
            self.data = np.array(data, dtype=dtype)
            self.dtype = self.data.dtype
            self.shape = self.data.shape
            self.ndim = self.data.ndim

        def fill(self, val):
            self.data.fill(val)

        def astype(self, dtype, copy=False):
            return MockCupyArray(self.data.astype(dtype, copy=copy))

        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = shape[0]
            return MockCupyArray(self.data.reshape(*shape))

        def transpose(self, *axes):
            if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
                axes = axes[0]
            return MockCupyArray(self.data.transpose(*axes))

        @property
        def T(self):
            return MockCupyArray(self.data.T)

        def __matmul__(self, other):
            other_data = other.data if isinstance(other, MockCupyArray) else other
            return MockCupyArray(self.data @ other_data)

        def __add__(self, other):
            other_data = other.data if isinstance(other, MockCupyArray) else other
            return MockCupyArray(self.data + other_data)

        def __iadd__(self, other):
            other_data = other.data if isinstance(other, MockCupyArray) else other
            self.data += other_data
            return self

        def __mul__(self, other):
            other_data = other.data if isinstance(other, MockCupyArray) else other
            return MockCupyArray(self.data * other_data)

        def __gt__(self, other):
            return MockCupyArray(self.data > other)

        def sum(self, axis=None):
            return MockCupyArray(self.data.sum(axis=axis))

        def mean(self, axis=None):
            return MockCupyArray(self.data.mean(axis=axis))

        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                idx = tuple(i.data if isinstance(i, MockCupyArray) else i for i in idx)
            elif isinstance(idx, MockCupyArray):
                idx = idx.data
            return MockCupyArray(self.data[idx])

        def __setitem__(self, idx, val):
            if isinstance(idx, tuple):
                idx = tuple(i.data if isinstance(i, MockCupyArray) else i for i in idx)
            elif isinstance(idx, MockCupyArray):
                idx = idx.data
            val_data = val.data if isinstance(val, MockCupyArray) else val
            self.data[idx] = val_data

    mock_cp.ndarray = MockCupyArray
    mock_cp.asarray.side_effect = lambda a, **kwargs: (
        a if isinstance(a, MockCupyArray) else MockCupyArray(a, dtype=kwargs.get("dtype"))
    )
    mock_cp.asnumpy.side_effect = lambda a: np.asarray(a.data) if isinstance(a, MockCupyArray) else np.asarray(a)
    mock_cp.zeros_like.side_effect = lambda a: MockCupyArray(np.zeros_like(a.data))
    mock_cp.zeros.side_effect = lambda shape, dtype=None: MockCupyArray(np.zeros(shape, dtype=dtype))
    mock_cp.empty.side_effect = lambda shape, dtype=None: MockCupyArray(np.empty(shape, dtype=dtype))
    mock_cp.dot.side_effect = lambda a, b: MockCupyArray(np.dot(a.data if isinstance(a, MockCupyArray) else a, b.data if isinstance(b, MockCupyArray) else b))
    mock_cp.sum.side_effect = lambda a, axis=None: MockCupyArray(np.sum(a.data if isinstance(a, MockCupyArray) else a, axis=axis))
    mock_cp.mean.side_effect = lambda a, axis=None: MockCupyArray(np.mean(a.data if isinstance(a, MockCupyArray) else a, axis=axis))
    mock_cp.maximum.side_effect = lambda a, b: MockCupyArray(np.maximum(a.data if isinstance(a, MockCupyArray) else a, b))
    mock_cp.pad.side_effect = lambda a, pad_width, mode="constant", **kwargs: MockCupyArray(np.pad(a.data, pad_width, mode=mode, **kwargs))
    mock_cp.arange.side_effect = lambda *args: MockCupyArray(np.arange(*args))
    mock_cp.argmax.side_effect = lambda a, axis=None: MockCupyArray(np.argmax(a.data if isinstance(a, MockCupyArray) else a, axis=axis))
    mock_cp.tile.side_effect = lambda a, reps: MockCupyArray(np.tile(a.data if isinstance(a, MockCupyArray) else a, reps))
    mock_cp.concatenate.side_effect = lambda items, axis=0: MockCupyArray(np.concatenate([i.data for i in items], axis=axis))
    mock_cp.split.side_effect = lambda ary, indices, axis=0: [MockCupyArray(s) for s in np.split(ary.data, indices, axis=axis)]
    mock_cp.inf = np.inf

    mock_add = MagicMock()
    mock_add.at = lambda a, indices, b: np.add.at(
        a.data,
        tuple(i.data if isinstance(i, MockCupyArray) else i for i in indices) if isinstance(indices, tuple) else (indices.data if isinstance(indices, MockCupyArray) else indices),
        b.data if isinstance(b, MockCupyArray) else b,
    )
    mock_cp.add = mock_add

    with patch.dict("sys.modules", {"cupy": mock_cp}):
        set_backend("gpu")

        net = Network(
            name="mock_net",
            model=Sequential(
                [
                    Linear(4, 3),
                    ReLU(),
                    Linear(3, 2),
                ]
            ),
            input_dim=4,
            output_dim=2,
        )

        net.to(device="gpu")
        for p in net.params():
            assert isinstance(p.data, MockCupyArray)
            assert isinstance(p.grad, MockCupyArray)

        x = to_device(np.random.randn(2, 4), device="gpu")
        out = net.forward(x, cache=True)
        assert isinstance(out, MockCupyArray)

        d_out = to_device(np.ones((2, 2)), device="gpu")
        d_x = net.backward(d_out)
        assert isinstance(d_x, MockCupyArray)

        # Move back to CPU
        net.to(device="cpu")
        for p in net.params():
            assert isinstance(p.data, np.ndarray)
            assert isinstance(p.grad, np.ndarray)


def test_optimizers_cpu_and_to_device():
    """Verify SGD and ADAM behavior and .to(device, dtype) migration."""
    p1 = Parameter.from_data(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    p1.grad = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # 1. SGD step
    sgd = SGD(learning_rate=0.1)
    sgd.parameters = [p1]
    sgd.step()
    np.testing.assert_allclose(p1.data, np.array([0.99, 1.98, 2.97]))
    assert sgd.to(device="cpu", dtype=np.float32) is sgd

    # 2. ADAM step and moments
    p2 = Parameter.from_data(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    p2.grad = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    adam = ADAM(learning_rate=0.01)
    adam.parameters = [p2]
    adam.step()
    assert len(adam.moment_1) == 1
    assert len(adam.moment_2) == 1
    assert adam.moment_1[0].dtype == np.float64
    assert adam.moment_2[0].dtype == np.float64

    # Convert ADAM moments to float32
    adam.to(dtype=np.float32)
    assert adam.moment_1[0].dtype == np.float32
    assert adam.moment_2[0].dtype == np.float32


def test_loss_and_metrics_backend_dispatch():
    """Verify softmax_cross_entropy_loss, accuracy, and dqn_mse in float32 and float64."""
    # 1. Cross entropy loss & accuracy
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float32)
    y_true = np.array([0, 1], dtype=np.int64)

    loss, grad = softmax_cross_entropy_loss(logits, y_true)
    assert isinstance(loss, (float, np.floating))
    assert grad.dtype == np.float32
    assert grad.shape == logits.shape

    acc = accuracy(logits, y_true)
    assert isinstance(acc, (float, np.floating))
    assert acc == 1.0

    # 2. DQN MSE
    q_pred = np.array([[1.5, 2.0], [0.5, 3.0]], dtype=np.float32)
    actions = np.array([1, 0], dtype=np.int64)
    targets = np.array([2.5, 1.0], dtype=np.float32)

    loss_dqn, grad_dqn = dqn_mse(q_pred, (actions, targets))
    assert isinstance(loss_dqn, (float, np.floating))
    assert grad_dqn.dtype == np.float32
    assert grad_dqn.shape == q_pred.shape
    assert grad_dqn[0, 0] == 0.0
    assert grad_dqn[0, 1] != 0.0


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy/GPU is not available")
def test_real_gpu_optimizers_and_losses():
    """Verify numerical parity of optimizers and loss functions on GPU."""
    import cupy as cp

    # ADAM step parity
    p_cpu = Parameter.from_data(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    p_cpu.grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    adam_cpu = ADAM(learning_rate=0.01)
    adam_cpu.parameters = [p_cpu]
    adam_cpu.step()

    p_gpu = Parameter.from_data(to_device(np.array([1.0, 2.0, 3.0], dtype=np.float32), device="gpu"))
    p_gpu.grad = to_device(np.array([0.1, 0.2, 0.3], dtype=np.float32), device="gpu")
    adam_gpu = ADAM(learning_rate=0.01)
    adam_gpu.parameters = [p_gpu]
    adam_gpu.step()

    assert isinstance(p_gpu.data, cp.ndarray)
    np.testing.assert_allclose(to_numpy(p_gpu.data), p_cpu.data, rtol=1e-5, atol=1e-5)

    # Loss parity
    logits_cpu = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float32)
    y_true_cpu = np.array([0, 1], dtype=np.int64)
    loss_cpu, grad_cpu = softmax_cross_entropy_loss(logits_cpu, y_true_cpu)

    logits_gpu = to_device(logits_cpu, device="gpu")
    y_true_gpu = to_device(y_true_cpu, device="gpu")
    loss_gpu, grad_gpu = softmax_cross_entropy_loss(logits_gpu, y_true_gpu)

    assert isinstance(loss_gpu, float)
    assert isinstance(grad_gpu, cp.ndarray)
    np.testing.assert_allclose(loss_gpu, loss_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(grad_gpu), grad_cpu, rtol=1e-5, atol=1e-5)
