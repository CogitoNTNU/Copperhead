import numpy as np
import pytest
from coppercore.backend import is_gpu_available
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
from coppercore.model_driven_nn.construction.loader import Loader
from coppercore.model_driven_nn.network import Network


def test_save_and_load_model_directory(tmp_path):
    branch1 = Sequential([Linear(8, 4), ReLU()])
    branch2 = Sequential([Linear(8, 6), ReLU()])
    concat = Concat([branch1, branch2])
    model = Sequential([concat, Linear(10, 3)])
    net = Network("model_bundle", model, 8, 3)

    model_dir = tmp_path / "saved_model"
    net.save_model(model_dir)

    assert (model_dir / "manifest.json").exists()
    assert (model_dir / "model.yaml").exists()
    assert (model_dir / "weights.npz").exists()

    restored = Network.load_model(model_dir)

    assert restored.name == net.name
    assert restored.input_dim == net.input_dim
    assert restored.output_dim == net.output_dim
    assert restored.number() == net.number()

    x = np.random.randn(4, 8)
    np.testing.assert_allclose(net.predict(x), restored.predict(x))


def test_loader_load_model_method(tmp_path):
    model = Sequential([Linear(6, 4), ReLU(), Linear(4, 2)])
    net = Network("loader_bundle", model, 6, 2)

    model_dir = tmp_path / "saved_model_loader"
    net.save_model(model_dir)

    restored = Loader().load_model(model_dir)

    assert restored.name == net.name
    x = np.random.randn(3, 6)
    np.testing.assert_allclose(net.predict(x), restored.predict(x))


def test_save_model_overwrite_existing_directory(tmp_path):
    model1 = Sequential([Linear(4, 2)])
    net1 = Network("net_v1", model1, 4, 2)

    model_dir = tmp_path / "overwrite_dir"
    net1.save_model(model_dir)

    model2 = Sequential([Linear(4, 3), ReLU(), Linear(3, 2)])
    net2 = Network("net_v2", model2, 4, 2)
    net2.save_model(model_dir)

    restored = Network.load_model(model_dir)
    assert restored.name == net2.name
    x = np.random.randn(2, 4)
    np.testing.assert_allclose(net2.predict(x), restored.predict(x))


def test_conv_architecture_serialization_round_trip(tmp_path):
    model = Sequential(
        [
            Conv2D(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
                input_shape=(1, 8, 8),
            ),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Linear(4 * 4 * 4, 2),
        ]
    )
    net = Network("conv_roundtrip", model, 64, 2)

    model_dir = tmp_path / "conv_model"
    net.save_model(model_dir)

    restored = Network.load_model(model_dir)

    # 1. Correct architecture and parameter names
    orig_param_names = [name for name, _ in net.named_parameters()]
    rest_param_names = [name for name, _ in restored.named_parameters()]
    assert rest_param_names == orig_param_names
    assert rest_param_names == ["model.0.W", "model.0.b", "model.4.W", "model.4.b"]

    # 2. Identical saved and loaded weights
    orig_params = net.params()
    rest_params = restored.params()
    assert len(orig_params) == len(rest_params)
    for p_orig, p_rest in zip(orig_params, rest_params):
        np.testing.assert_array_equal(p_orig.data, p_rest.data)

    # 3. Matching predictions for the same input
    x = np.random.randn(2, 64)
    np.testing.assert_allclose(net.predict(x), restored.predict(x))


def test_avgpool2d_reconstruction_from_config(tmp_path):
    model = Sequential(
        [
            AvgPool2D(kernel_size=2, stride=2, padding=0, input_shape=(1, 6, 6)),
            Flatten(),
            Linear(1 * 3 * 3, 2),
        ]
    )
    net = Network("avgpool_roundtrip", model, 36, 2)

    model_dir = tmp_path / "avgpool_model"
    net.save_model(model_dir)

    restored = Network.load_model(model_dir)
    assert isinstance(restored.model.components[0], AvgPool2D)
    assert restored.model.components[0].kernel_size == (2, 2)
    assert restored.model.components[0].stride == (2, 2)
    assert restored.model.components[0].padding == (0, 0)

    x = np.random.randn(2, 36)
    np.testing.assert_allclose(net.predict(x), restored.predict(x))


def test_model_dtype_preservation(tmp_path):
    # Float32 model
    net32 = Network("net32", Sequential([Linear(4, 2)]), 4, 2).to(dtype=np.float32)
    net32.save_model(tmp_path / "model_f32")
    loaded32 = Network.load_model(tmp_path / "model_f32")
    assert loaded32.params()[0].data.dtype == np.float32
    assert loaded32.params()[0].grad.dtype == np.float32

    # Float64 model
    net64 = Network("net64", Sequential([Linear(4, 2)]), 4, 2).to(dtype=np.float64)
    net64.save_model(tmp_path / "model_f64")
    loaded64 = Network.load_model(tmp_path / "model_f64")
    assert loaded64.params()[0].data.dtype == np.float64
    assert loaded64.params()[0].grad.dtype == np.float64

    # Explicit dtype conversion on load
    loaded_f32_from_f64 = Network.load_model(
        tmp_path / "model_f64", device="cpu", dtype=np.float32
    )
    assert loaded_f32_from_f64.params()[0].data.dtype == np.float32
    assert loaded_f32_from_f64.params()[0].grad.dtype == np.float32


def test_gpu_model_save_and_load(tmp_path):
    if not is_gpu_available():
        pytest.skip("GPU / CuPy hardware not available")

    net_gpu = Network("net_gpu", Sequential([Linear(4, 2)]), 4, 2).to(
        device="gpu", dtype=np.float32
    )
    net_gpu.save_model(tmp_path / "gpu_model")

    # Restore onto CPU
    restored_cpu = Network.load_model(tmp_path / "gpu_model", device="cpu")
    assert isinstance(restored_cpu.params()[0].data, np.ndarray)
    assert restored_cpu.params()[0].data.dtype == np.float32

    # Restore onto GPU
    restored_gpu = Network.load_model(tmp_path / "gpu_model", device="gpu")
    assert hasattr(restored_gpu.params()[0].data, "get")
    assert restored_gpu.params()[0].data.dtype == np.float32
