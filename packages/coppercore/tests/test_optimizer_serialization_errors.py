from unittest.mock import MagicMock

import numpy as np
import pytest
from coppercore.model_driven_nn.components import Linear, Sequential
from coppercore.model_driven_nn.network import Network
from coppercore.model_driven_nn.parameter import Parameter
from coppercore.optimizer.adam import ADAM
from coppercore.optimizer.sgd import SGD


def test_unbound_optimizer_state_dict_fails():
    model = Sequential([Linear(2, 2)])
    net = Network("net", model, 2, 2)
    opt = ADAM()
    with pytest.raises(RuntimeError, match="Optimizer parameters have not been set"):
        opt.state_dict(net)


def test_unmatched_parameters_between_optimizer_and_network_fails():
    net1 = Network("net1", Sequential([Linear(2, 2)]), 2, 2)
    net2 = Network("net2", Sequential([Linear(2, 2)]), 2, 2)

    opt = ADAM()
    opt.parameters = net1.params()

    with pytest.raises(
        ValueError, match="Optimizer parameters do not match network parameters"
    ):
        opt.state_dict(net2)


def test_network_duplicate_parameter_names_fails():
    p1 = Parameter.from_data(np.ones((2, 2)))
    p2 = Parameter.from_data(np.ones((2, 2)))

    fake_net = MagicMock(spec=Network)
    fake_net.named_parameters.return_value = [("model.0.W", p1), ("model.0.W", p2)]

    opt = ADAM()
    opt.parameters = [p1, p2]

    with pytest.raises(ValueError, match="Network contains duplicate parameter names"):
        opt.state_dict(fake_net)


def test_network_duplicate_parameter_objects_fails():
    p1 = Parameter.from_data(np.ones((2, 2)))

    fake_net = MagicMock(spec=Network)
    fake_net.named_parameters.return_value = [("model.0.W", p1), ("model.0.b", p1)]

    opt = ADAM()
    opt.parameters = [p1, Parameter.from_data(np.ones((2, 2)))]

    with pytest.raises(
        ValueError, match="Network contains duplicate parameter objects"
    ):
        opt.state_dict(fake_net)


def test_optimizer_duplicate_parameter_objects_fails():
    p1 = Parameter.from_data(np.ones((2, 2)))
    p2 = Parameter.from_data(np.ones((2, 2)))

    fake_net = MagicMock(spec=Network)
    fake_net.named_parameters.return_value = [("model.0.W", p1), ("model.0.b", p2)]

    opt = ADAM()
    opt.parameters = [p1, p1]

    with pytest.raises(
        ValueError, match="Optimizer contains duplicate parameter objects"
    ):
        opt.state_dict(fake_net)


def test_optimizer_type_mismatches():
    model = Sequential([Linear(2, 2)])
    net = Network("net", model, 2, 2)

    adam = ADAM()
    adam.parameters = net.params()
    adam_state = adam.state_dict(net)

    sgd = SGD(learning_rate=0.01)
    sgd.parameters = net.params()
    sgd_state = sgd.state_dict(net)

    # Loading Adam state into SGD must fail
    with pytest.raises(ValueError, match="Checkpoint is not an SGD optimizer"):
        sgd.load_state_dict(adam_state, net)

    # Loading SGD state into ADAM must fail
    with pytest.raises(ValueError, match="Checkpoint is not an Adam optimizer"):
        adam.load_state_dict(sgd_state, net)


def test_checkpoint_parameters_mismatch_fails():
    model = Sequential([Linear(2, 2)])
    net = Network("net", model, 2, 2)

    opt = ADAM()
    opt.parameters = net.params()

    invalid_state = {
        "optimizer_type": "ADAM",
        "learning_rate": 0.001,
        "moment_1_decay": 0.9,
        "moment_2_decay": 0.999,
        "eps": 1e-8,
        "time_step": 0,
        "state": {
            "unrelated.layer.W": {"moment_1": None, "moment_2": None},
        },
    }

    with pytest.raises(ValueError, match="Checkpoint parameters do not match network"):
        opt.load_state_dict(invalid_state, net)


def test_corrupt_file_fails(tmp_path):
    model = Sequential([Linear(2, 2)])
    net = Network("net", model, 2, 2)

    corrupt_file = tmp_path / "corrupt.npz"
    corrupt_file.write_text("not a valid npz file")

    opt = ADAM()
    opt.parameters = net.params()

    with pytest.raises(Exception):
        opt.load_state(corrupt_file, net)
