import json

import numpy as np
import pytest
from coppercore.model_driven_nn.components import Linear, Sequential
from coppercore.model_driven_nn.components.base import Component
from coppercore.model_driven_nn.network import Network


def test_missing_weights_file():
    model = Sequential([Linear(4, 2)])
    net = Network("test_net", model, 4, 2)

    with pytest.raises(FileNotFoundError, match="Weight file not found"):
        net.load_weights("non_existent_weights.npz")


def test_missing_state_dict_key():
    model = Sequential([Linear(4, 2)])
    net = Network("test_net", model, 4, 2)

    state = net.state_dict()
    del state["model.0.W"]

    with pytest.raises(ValueError, match="missing keys: \\['model.0.W'\\]"):
        net.load_state_dict(state)


def test_extra_state_dict_key():
    model = Sequential([Linear(4, 2)])
    net = Network("test_net", model, 4, 2)

    state = net.state_dict()
    state["model.extra.param"] = np.zeros((2, 2))

    with pytest.raises(ValueError, match="unexpected keys: \\['model.extra.param'\\]"):
        net.load_state_dict(state)


def test_wrong_parameter_shape():
    model = Sequential([Linear(4, 2)])
    net = Network("test_net", model, 4, 2)

    state = net.state_dict()
    state["model.0.W"] = np.zeros((4, 3))  # expected (4, 2)

    with pytest.raises(ValueError, match="Shape mismatch for parameter 'model.0.W'"):
        net.load_state_dict(state)


def test_non_numeric_parameter():
    model = Sequential([Linear(4, 2)])
    net = Network("test_net", model, 4, 2)

    state = net.state_dict()
    state["model.0.b"] = np.array(["a", "b"])

    with pytest.raises(ValueError, match="must have a numeric dtype"):
        net.load_state_dict(state)


def test_corrupt_npz_archive(tmp_path):
    corrupt_file = tmp_path / "corrupt.npz"
    corrupt_file.write_bytes(b"not a valid zip or npz file")

    model = Sequential([Linear(4, 2)])
    net = Network("test_net", model, 4, 2)

    with pytest.raises(ValueError, match="Corrupt or invalid weight file"):
        net.load_weights(corrupt_file)


def test_unsupported_component_serialization():
    class DummyComponent(Component):
        def forward(self, x, *, cache=True):
            return x

        def backward(self, d_y):
            return d_y

    model = Sequential([DummyComponent()])
    net = Network("dummy_net", model, 4, 4)

    with pytest.raises(NotImplementedError, match="does not support serialization"):
        net.to_config()


def test_missing_model_directory():
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        Network.load_model("non_existent_dir_12345")


def test_missing_manifest_file(tmp_path):
    empty_dir = tmp_path / "empty_model"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        Network.load_model(empty_dir)


def test_invalid_manifest_json(tmp_path):
    model_dir = tmp_path / "bad_json_model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text("invalid json content", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid manifest JSON"):
        Network.load_model(model_dir)


def test_unsupported_manifest_version(tmp_path):
    model_dir = tmp_path / "future_version_model"
    model_dir.mkdir()
    manifest = {
        "format_version": 999,
        "architecture": "model.yaml",
        "weights": "weights.npz",
    }
    with (model_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with pytest.raises(
        ValueError, match="Unsupported serialization format version: 999"
    ):
        Network.load_model(model_dir)


def test_missing_architecture_in_manifest(tmp_path):
    model_dir = tmp_path / "missing_arch_model"
    model_dir.mkdir()
    manifest = {
        "format_version": 1,
        "architecture": "non_existent.yaml",
        "weights": "weights.npz",
    }
    with (model_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with pytest.raises(
        FileNotFoundError, match="Architecture file specified in manifest not found"
    ):
        Network.load_model(model_dir)


def test_missing_weights_in_manifest(tmp_path):
    model = Sequential([Linear(4, 2)])
    net = Network("test_net", model, 4, 2)

    model_dir = tmp_path / "missing_weights_model"
    model_dir.mkdir()
    net.save_architecture(model_dir / "model.yaml")

    manifest = {
        "format_version": 1,
        "architecture": "model.yaml",
        "weights": "missing_weights.npz",
    }
    with (model_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with pytest.raises(
        FileNotFoundError, match="Weight file specified in manifest not found"
    ):
        Network.load_model(model_dir)
