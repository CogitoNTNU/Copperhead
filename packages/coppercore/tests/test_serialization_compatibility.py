from pathlib import Path
import numpy as np

from coppercore.model_driven_nn.construction.loader import Loader
from coppercore.model_driven_nn.network import Network


def test_v1_yaml_compatibility(tmp_path):
    v1_path = Path("projects/even/networks/v1.yaml")
    assert v1_path.exists(), "v1.yaml must exist"

    net = Loader().load(v1_path)

    exported_yaml = tmp_path / "v1_exported.yaml"
    net.save_architecture(exported_yaml)

    restored = Loader().load(exported_yaml)

    assert restored.name == net.name
    assert restored.input_dim == net.input_dim
    assert restored.output_dim == net.output_dim
    assert restored.number() == net.number()
    assert len(restored.params()) == len(net.params())

    for p_orig, p_rest in zip(net.params(), restored.params()):
        assert p_orig.data.shape == p_rest.data.shape


def test_definition_of_done_workflow(tmp_path):
    v1_path = Path("projects/even/networks/v1.yaml")
    network = Loader().load(v1_path)

    # Simulate training by mutating parameters
    for param in network.params():
        param.data += np.random.randn(*param.data.shape) * 0.1

    save_path = tmp_path / "models" / "trained-network"
    network.save_model(save_path)

    restored = Network.load_model(save_path)

    x = np.random.randn(2, 784)
    np.testing.assert_allclose(
        network.predict(x),
        restored.predict(x),
    )


def test_forward_backward_params_zero_grad_lifecycle():
    v1_path = Path("projects/even/networks/v1.yaml")
    network = Loader().load(v1_path)

    x = np.random.randn(2, 784)
    out = network.forward(x)
    assert out.shape == (2, 10)

    d_y = np.ones_like(out)
    d_x = network.backward(d_y)
    assert d_x.shape == (2, 784)

    # Check gradients are non-zero
    assert any(np.abs(p.grad).sum() > 0 for p in network.params())

    network.zero_grad()
    for p in network.params():
        assert np.all(p.grad == 0)
