import numpy as np

from coppercore.model_driven_nn.components import Concat, Linear, ReLU, Sequential
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
