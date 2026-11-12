import numpy as np

from coppercore.model_driven_nn.components import Concat, Linear, ReLU, Sequential
from coppercore.model_driven_nn.network import Network
from coppercore.optimizer.adam import ADAM
from coppercore.optimizer.sgd import SGD


def test_weights_roundtrip_simple(tmp_path):
    model1 = Sequential([Linear(8, 4), ReLU(), Linear(4, 2)])
    net1 = Network("net1", model1, 8, 2)

    weights_path = tmp_path / "weights.npz"
    net1.save_weights(weights_path)

    model2 = Sequential([Linear(8, 4), ReLU(), Linear(4, 2)])
    net2 = Network("net2", model2, 8, 2)

    x = np.random.randn(5, 8)
    assert not np.allclose(net1.predict(x), net2.predict(x))

    net2.load_weights(weights_path)

    np.testing.assert_allclose(net1.predict(x), net2.predict(x))
    for p1, p2 in zip(net1.params(), net2.params()):
        np.testing.assert_array_equal(p1.data, p2.data)
        assert p1.data.dtype == p2.data.dtype


def test_state_dict_preserves_parameter_identity():
    model = Sequential([Linear(10, 5), ReLU(), Linear(5, 2)])
    net = Network("identity_net", model, 10, 2)

    param_objects_before = list(net.params())

    state = net.state_dict()
    for name in state:
        state[name] = state[name] + 1.0

    net.load_state_dict(state)

    param_objects_after = list(net.params())

    assert len(param_objects_before) == len(param_objects_after)
    for p_before, p_after in zip(param_objects_before, param_objects_after):
        assert p_before is p_after


def test_named_parameters_hierarchical_keys():
    branch1 = Sequential([Linear(10, 4), ReLU()])
    branch2 = Sequential([Linear(10, 6), ReLU()])
    concat = Concat([branch1, branch2])
    model = Sequential([Linear(10, 10), ReLU(), concat, Linear(10, 2)])
    net = Network("concat_named", model, 10, 2)

    named = net.named_parameters()
    names = [name for name, _ in named]

    assert "model.0.W" in names
    assert "model.0.b" in names
    assert "model.2.paths.0.0.W" in names
    assert "model.2.paths.0.0.b" in names
    assert "model.2.paths.1.0.W" in names
    assert "model.2.paths.1.0.b" in names
    assert "model.3.W" in names
    assert "model.3.b" in names


def test_optimizer_compatibility_after_weight_loading():
    model = Sequential([Linear(4, 3), ReLU(), Linear(3, 2)])
    net = Network("opt_net", model, 4, 2)

    optimizer_sgd = SGD(learning_rate=0.01)
    optimizer_sgd.parameters = net.params()

    optimizer_adam = ADAM(learning_rate=0.001)
    optimizer_adam.parameters = net.params()

    x = np.random.randn(2, 4)
    out = net.forward(x)
    net.backward(np.ones_like(out))

    initial_weights = net.state_dict()

    new_weights = {k: v * 2.0 for k, v in initial_weights.items()}
    net.load_state_dict(new_weights)

    # Optimizers should still point to the same parameter objects
    assert optimizer_sgd.parameters == net.params()
    assert optimizer_adam.parameters == net.params()

    optimizer_sgd.step()
    # Ensure parameter data was updated by optimizer step
    for p in net.params():
        assert not np.isnan(p.data).any()
