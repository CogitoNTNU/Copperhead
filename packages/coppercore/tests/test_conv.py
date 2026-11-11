import tempfile
from pathlib import Path

import numpy as np
import pytest
from coppercore.cli.flattening import flatten_yaml
from coppercore.cli.graphing import graph_yaml
from coppercore.model_driven_nn import Network
from coppercore.model_driven_nn.components import (
    AvgPool2D,
    Conv2D,
    Flatten,
    MaxPool2D,
    Sequential,
)
from coppercore.model_driven_nn.construction.builder import Builder
from coppercore.model_driven_nn.construction.loader import Loader
from coppercore.model_driven_nn.construction.parser import Parser
from coppercore.model_driven_nn.construction.validator import Validator


def test_conv2d_init():
    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
    assert conv.in_channels == 1
    assert conv.out_channels == 4
    assert conv.kernel_size == (3, 3)
    assert conv.stride == (1, 1)
    assert conv.padding == (1, 1)
    assert len(conv.params()) == 2
    assert conv.W.data.shape == (4, 1, 3, 3)
    assert conv.b is not None and conv.b.data.shape == (4,)


def test_conv2d_no_bias():
    conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, bias=False)
    assert conv.b is None
    assert len(conv.params()) == 1
    x = np.random.randn(2, 2, 5, 5)
    out = conv.forward(x)
    assert out.shape == (2, 4, 3, 3)
    d_x = conv.backward(np.ones_like(out))
    assert d_x.shape == x.shape


def test_conv2d_invalid_init():
    with pytest.raises(ValueError, match="in_channels must be positive"):
        Conv2D(in_channels=0, out_channels=4)

    with pytest.raises(ValueError, match="out_channels must be positive"):
        Conv2D(in_channels=1, out_channels=-1)

    with pytest.raises(ValueError, match="kernel_size must be positive"):
        Conv2D(in_channels=1, out_channels=4, kernel_size=0)

    with pytest.raises(ValueError, match="stride must be positive"):
        Conv2D(in_channels=1, out_channels=4, stride=-1)

    with pytest.raises(ValueError, match="input_shape must have 2 or 3 elements"):
        Conv2D(in_channels=1, out_channels=4, input_shape=[1, 2, 3, 4])


def test_conv2d_forward_shapes():
    # 4D input: (N, C, H, W) = (2, 1, 8, 8)
    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
    x = np.random.randn(2, 1, 8, 8)
    out = conv.forward(x)
    assert out.shape == (2, 4, 4, 4)

    # 2D flattened input: (2, 64) with inferred square shape
    x_2d = x.reshape(2, 64)
    out_2d = conv.forward(x_2d)
    assert out_2d.shape == (2, 4, 4, 4)


def test_conv2d_backward_before_forward():
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3)
    with pytest.raises(RuntimeError, match="Backward called before forward"):
        conv.backward(np.ones((1, 2, 6, 6)))


def test_conv2d_gradient_check():
    """Verify gradients of Conv2D using finite difference approximations."""
    rng = np.random.default_rng(42)
    n, c_in, h_in, w_in = 2, 2, 5, 5
    c_out = 3
    k_h, k_w = 3, 3
    stride = 1
    padding = 1

    conv = Conv2D(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=(k_h, k_w),
        stride=stride,
        padding=padding,
        seed=123,
    )

    x = rng.standard_normal((n, c_in, h_in, w_in))

    # Analytical gradients
    conv.zero_grad()
    out = conv.forward(x, cache=True)
    d_out = rng.standard_normal(out.shape)
    d_x = conv.backward(d_out)

    # Check d_x via finite difference
    eps = 1e-6
    d_x_num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_pos = x.copy()
        x_pos[idx] += eps
        out_pos = conv.forward(x_pos, cache=False)

        x_neg = x.copy()
        x_neg[idx] -= eps
        out_neg = conv.forward(x_neg, cache=False)

        d_x_num[idx] = np.sum((out_pos - out_neg) / (2 * eps) * d_out)

    rel_error_x = np.linalg.norm(d_x - d_x_num) / (
        np.linalg.norm(d_x) + np.linalg.norm(d_x_num) + 1e-12
    )
    assert rel_error_x < 1e-5

    # Check d_W via finite difference
    d_w_num = np.zeros_like(conv.W.data)
    for idx in np.ndindex(conv.W.data.shape):
        orig_val = conv.W.data[idx]

        conv.W.data[idx] = orig_val + eps
        out_pos = conv.forward(x, cache=False)

        conv.W.data[idx] = orig_val - eps
        out_neg = conv.forward(x, cache=False)

        conv.W.data[idx] = orig_val
        d_w_num[idx] = np.sum((out_pos - out_neg) / (2 * eps) * d_out)

    rel_error_w = np.linalg.norm(conv.W.grad - d_w_num) / (
        np.linalg.norm(conv.W.grad) + np.linalg.norm(d_w_num) + 1e-12
    )
    assert rel_error_w < 1e-5

    # Check d_b via finite difference
    assert conv.b is not None
    d_b_num = np.zeros_like(conv.b.data)
    for idx in np.ndindex(conv.b.data.shape):
        orig_val = conv.b.data[idx]

        conv.b.data[idx] = orig_val + eps
        out_pos = conv.forward(x, cache=False)

        conv.b.data[idx] = orig_val - eps
        out_neg = conv.forward(x, cache=False)

        conv.b.data[idx] = orig_val
        d_b_num[idx] = np.sum((out_pos - out_neg) / (2 * eps) * d_out)

    rel_error_b = np.linalg.norm(conv.b.grad - d_b_num) / (
        np.linalg.norm(conv.b.grad) + np.linalg.norm(d_b_num) + 1e-12
    )
    assert rel_error_b < 1e-5


def test_flatten_layer():
    flatten = Flatten()
    x = np.random.randn(3, 4, 5, 6)
    out = flatten.forward(x)
    assert out.shape == (3, 120)

    d_y = np.ones_like(out)
    d_x = flatten.backward(d_y)
    assert d_x.shape == x.shape
    assert np.allclose(d_x, 1.0)
    assert flatten.params() == []


def test_maxpool2d_gradient_check():
    rng = np.random.default_rng(42)
    pool = MaxPool2D(kernel_size=2, stride=2, padding=0)
    x = rng.standard_normal((2, 3, 6, 6))

    out = pool.forward(x, cache=True)
    assert out.shape == (2, 3, 3, 3)

    d_out = rng.standard_normal(out.shape)
    d_x = pool.backward(d_out)
    assert d_x.shape == x.shape

    eps = 1e-6
    d_x_num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_pos = x.copy()
        x_pos[idx] += eps
        out_pos = pool.forward(x_pos, cache=False)

        x_neg = x.copy()
        x_neg[idx] -= eps
        out_neg = pool.forward(x_neg, cache=False)

        d_x_num[idx] = np.sum((out_pos - out_neg) / (2 * eps) * d_out)

    rel_error = np.linalg.norm(d_x - d_x_num) / (
        np.linalg.norm(d_x) + np.linalg.norm(d_x_num) + 1e-12
    )
    assert rel_error < 1e-5


def test_avgpool2d_gradient_check():
    rng = np.random.default_rng(42)
    pool = AvgPool2D(kernel_size=2, stride=2, padding=0)
    x = rng.standard_normal((2, 3, 6, 6))

    out = pool.forward(x, cache=True)
    assert out.shape == (2, 3, 3, 3)

    d_out = rng.standard_normal(out.shape)
    d_x = pool.backward(d_out)
    assert d_x.shape == x.shape

    eps = 1e-6
    d_x_num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_pos = x.copy()
        x_pos[idx] += eps
        out_pos = pool.forward(x_pos, cache=False)

        x_neg = x.copy()
        x_neg[idx] -= eps
        out_neg = pool.forward(x_neg, cache=False)

        d_x_num[idx] = np.sum((out_pos - out_neg) / (2 * eps) * d_out)

    rel_error = np.linalg.norm(d_x - d_x_num) / (
        np.linalg.norm(d_x) + np.linalg.norm(d_x_num) + 1e-12
    )
    assert rel_error < 1e-5


def test_builder_cnn_shape_inference():
    spec = {
        "type": "sequential",
        "layers": [
            {
                "type": "conv2d",
                "in_channels": 1,
                "out_channels": 8,
                "kernel_size": 3,
                "padding": 1,
            },
            {"type": "relu"},
            {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
            {
                "type": "conv2d",
                "out_channels": 16,
                "kernel_size": 3,
                "padding": 1,
            },
            {"type": "relu"},
            {"type": "avgpool2d", "kernel_size": 2, "stride": 2},
            {"type": "flatten"},
            {"type": "linear", "output_dim": 10},
        ],
    }

    # input is 28x28 = 784
    seq, out_dim = Builder().build(spec, 784)
    assert isinstance(seq, Sequential)
    assert out_dim == 10

    # Layer 1: Conv2D(1 -> 8, 28x28) -> (8, 28, 28)
    # Layer 2: ReLU
    # Layer 3: MaxPool2D(2x2) -> (8, 14, 14)
    # Layer 4: Conv2D(in_channels inferred as 8, out 16) -> (16, 14, 14)
    # Layer 5: ReLU
    # Layer 6: AvgPool2D(2x2) -> (16, 7, 7) = 784 features
    # Layer 7: Flatten
    # Layer 8: Linear(784 -> 10)

    conv2 = seq.components[3]
    assert isinstance(conv2, Conv2D)
    assert conv2.in_channels == 8
    assert conv2.input_shape == (8, 14, 14)

    x = np.random.randn(4, 784)
    out = seq.forward(x)
    assert out.shape == (4, 10)

    d_x = seq.backward(np.ones_like(out))
    assert d_x.shape == x.shape


def test_cnn_yaml_loading_and_training():
    yaml_content = """
name: cnn_v1
input_dim: 64
output_dim: 2
model:
  type: sequential
  layers:
    - type: conv2d
      in_channels: 1
      out_channels: 4
      kernel_size: 3
      padding: 1
    - type: relu
    - type: maxpool2d
      kernel_size: 2
    - type: flatten
    - type: linear
      output_dim: 2
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = Path(f.name)

    try:
        network = Loader().load(path)
        assert isinstance(network, Network)
        assert network.name.startswith("cnn_v1")
        assert network.input_dim == 64
        assert network.output_dim == 2

        # 8x8 input = 64 features
        rng = np.random.default_rng(42)
        x = rng.standard_normal((8, 64)) * 0.1
        y = np.zeros((8, 2))
        y[np.arange(8), rng.integers(0, 2, size=8)] = 1.0

        # Initial prediction
        initial_pred = network.predict(x)
        initial_loss = np.mean((initial_pred - y) ** 2)

        # Training step
        lr = 0.01
        for _ in range(15):
            network.zero_grad()
            out = network.forward(x, cache=True)
            d_out = 2 * (out - y) / len(x)
            network.backward(d_out)
            for param in network.params():
                param.data -= lr * param.grad

        final_pred = network.predict(x)
        final_loss = np.mean((final_pred - y) ** 2)
        assert final_loss < initial_loss
    finally:
        path.unlink(missing_ok=True)


def test_validator_and_graphing_cnn():
    yaml_content = """
name: cnn_test
input_dim: 16
output_dim: 4
model:
  type: sequential
  layers:
    - type: conv2d
      in_channels: 1
      out_channels: 2
      kernel_size: 3
      padding: 1
    - type: relu
    - type: maxpool2d
      kernel_size: 2
    - type: flatten
    - type: linear
      output_dim: 4
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)

    dot_path = yaml_path.with_suffix(".dot")
    flat_path = yaml_path.with_suffix(".flat.yaml")

    try:
        arch = Parser().parse(yaml_path)
        validated = Validator().validate_architecture(arch)
        assert validated.name == "cnn_test"

        # Test CLI flattening and graphing
        flatten_yaml(yaml_path, destination=flat_path)
        assert flat_path.exists()
        graph_yaml(yaml_path, destination=dot_path)
        assert dot_path.exists()
        content = dot_path.read_text(encoding="utf-8")
        assert "Conv2D" in content
        assert "MaxPool2D" in content
        assert "Flatten" in content
        assert "Linear" in content
    finally:
        yaml_path.unlink(missing_ok=True)
        dot_path.unlink(missing_ok=True)
        flat_path.unlink(missing_ok=True)


def test_validator_conv_and_pool_errors():
    val = Validator()

    # Invalid bias type
    with pytest.raises(TypeError, match="bias must be a boolean"):
        val.validate_architecture(
            {
                "name": "bad_bias",
                "input_dim": 16,
                "model": {
                    "type": "sequential",
                    "layers": [
                        {"type": "conv2d", "out_channels": 4, "bias": "yes"},
                        {"type": "relu"},
                        {"type": "linear", "output_dim": 2},
                    ],
                },
            }
        )

    # Invalid padding (negative)
    with pytest.raises(ValueError, match="padding must be a non-negative integer"):
        val.validate_architecture(
            {
                "name": "bad_padding",
                "input_dim": 16,
                "model": {
                    "type": "sequential",
                    "layers": [
                        {"type": "conv2d", "out_channels": 4, "padding": -1},
                        {"type": "relu"},
                        {"type": "linear", "output_dim": 2},
                    ],
                },
            }
        )

    # Invalid kernel_size (zero)
    with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
        val.validate_architecture(
            {
                "name": "bad_pool_k",
                "input_dim": 16,
                "model": {
                    "type": "sequential",
                    "layers": [
                        {"type": "conv2d", "out_channels": 4},
                        {"type": "relu"},
                        {"type": "maxpool2d", "kernel_size": 0},
                        {"type": "linear", "output_dim": 2},
                    ],
                },
            }
        )


def test_conv2d_non_square_and_output_shape():
    conv = Conv2D(
        in_channels=2,
        out_channels=5,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 0),
        input_shape=(2, 6, 8),
    )
    assert conv.output_shape == (5, 3, 7)
    x = np.random.randn(3, 2, 6, 8)
    out = conv.forward(x)
    assert out.shape == (3, 5, 3, 7)


def test_builder_unified_dispatch():
    builder = Builder()

    # Linear layer
    linear, out_dim = builder.build({"type": "linear", "output_dim": 10}, 5)
    assert out_dim == 10

    # Conv layer with shape context
    conv, out_dim = builder.build(
        {"type": "conv2d", "out_channels": 4, "kernel_size": 3, "padding": 1},
        16,
        (1, 4, 4),
    )
    assert isinstance(conv, Conv2D)
    assert out_dim == 4 * 4 * 4

    # MaxPool layer with shape context
    pool, out_dim = builder.build(
        {"type": "maxpool2d", "kernel_size": 2},
        64,
        (4, 4, 4),
    )
    assert isinstance(pool, MaxPool2D)
    assert out_dim == 4 * 2 * 2
    assert pool.output_shape == (4, 2, 2)

    # AvgPool layer with shape context
    avg_pool, out_dim = builder.build(
        {"type": "avgpool2d", "kernel_size": 2},
        64,
        (4, 4, 4),
    )
    assert isinstance(avg_pool, AvgPool2D)
    assert out_dim == 4 * 2 * 2
    assert avg_pool.output_shape == (4, 2, 2)
