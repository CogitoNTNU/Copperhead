import numpy as np

from coppercore.model_driven_nn.components import Linear, Sequential
from coppercore.model_driven_nn.network import Network
from coppercore.optimizer.sgd import SGD


def test_sgd_state_dict_round_trip():
    model = Sequential([Linear(4, 2)])
    net = Network("sgd_test", model, 4, 2)

    opt = SGD(learning_rate=0.042)
    opt.parameters = net.params()

    state = opt.state_dict(net)
    assert state["optimizer_type"] == "SGD"
    assert state["learning_rate"] == 0.042

    opt2 = SGD(learning_rate=0.001)
    opt2.parameters = net.params()
    opt2.load_state_dict(state, net)

    assert opt2.learning_rate == 0.042


def test_sgd_file_save_load_round_trip(tmp_path):
    model = Sequential([Linear(4, 2)])
    net = Network("sgd_file_test", model, 4, 2)

    opt = SGD(learning_rate=0.035)
    opt.parameters = net.params()

    save_path = tmp_path / "sgd_saved.npz"
    opt.save_state(save_path, net)

    assert save_path.exists()

    opt_restored = SGD(learning_rate=0.01)
    opt_restored.parameters = net.params()
    opt_restored.load_state(save_path, net)

    assert opt_restored.learning_rate == 0.035


def test_sgd_parameter_identity_preservation():
    model = Sequential([Linear(4, 2)])
    net = Network("sgd_ident", model, 4, 2)

    params_before = net.params()
    param_ids_before = [id(p) for p in params_before]

    opt = SGD(learning_rate=0.05)
    opt.parameters = net.params()

    state = opt.state_dict(net)

    opt2 = SGD(learning_rate=0.01)
    opt2.parameters = net.params()
    opt2.load_state_dict(state, net)

    params_after = net.params()
    param_ids_after = [id(p) for p in params_after]

    assert param_ids_before == param_ids_after
    for p_opt, p_net in zip(opt2.parameters, net.params()):
        assert p_opt is p_net


def test_sgd_step_updates_weights():
    model = Sequential([Linear(2, 2)])
    net = Network("sgd_step_test", model, 2, 2)

    opt = SGD(learning_rate=0.1)
    opt.parameters = net.params()

    initial_w = net.params()[0].data.copy()
    net.zero_grad()
    out = net.forward(np.ones((1, 2)))
    net.backward(np.ones_like(out))

    opt.step()

    updated_w = net.params()[0].data
    assert not np.array_equal(initial_w, updated_w)


def test_serialization_helpers_with_lists(tmp_path):
    from coppercore.optimizer.serialization import (
        load_optimizer_state_from_file,
        save_optimizer_state_to_file,
    )

    state = {
        "optimizer_type": "CUSTOM",
        "nested_list": [1, [2, np.array([3.0, 4.0])]],
    }
    save_path = tmp_path / "custom.npz"
    save_optimizer_state_to_file(save_path, state)
    loaded = load_optimizer_state_from_file(save_path)

    assert loaded["optimizer_type"] == "CUSTOM"
    assert loaded["nested_list"][0] == 1
    assert loaded["nested_list"][1][0] == 2
    np.testing.assert_array_equal(loaded["nested_list"][1][1], np.array([3.0, 4.0]))
