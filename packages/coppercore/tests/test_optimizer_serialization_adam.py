import numpy as np
from coppercore.backend import get_array_module, is_gpu_available
from coppercore.model_driven_nn.components import Linear, ReLU, Sequential
from coppercore.model_driven_nn.network import Network
from coppercore.optimizer.adam import ADAM


def test_adam_state_dict_round_trip():
    model = Sequential([Linear(4, 3), ReLU(), Linear(3, 2)])
    net = Network("adam_test", model, 4, 2)

    opt = ADAM(learning_rate=0.002, moment_1_decay=0.85, moment_2_decay=0.95, eps=1e-7)
    opt.parameters = net.params()

    # Perform 2 steps to populate moments
    for _ in range(2):
        net.zero_grad()
        out = net.forward(np.ones((2, 4)))
        net.backward(np.ones_like(out))
        opt.step()

    state = opt.state_dict(net)

    assert state["optimizer_type"] == "ADAM"
    assert state["learning_rate"] == 0.002
    assert state["moment_1_decay"] == 0.85
    assert state["moment_2_decay"] == 0.95
    assert state["eps"] == 1e-7
    assert state["time_step"] == 2
    assert "model.0.W" in state["state"]
    assert "model.0.b" in state["state"]
    assert "model.2.W" in state["state"]
    assert "model.2.b" in state["state"]

    # Restore into new optimizer with new network
    net2 = Network(
        "adam_test_2", Sequential([Linear(4, 3), ReLU(), Linear(3, 2)]), 4, 2
    )
    opt2 = ADAM()
    opt2.parameters = net2.params()
    opt2.load_state_dict(state, net2)

    assert opt2.learning_rate == 0.002
    assert opt2.moment_1_decay == 0.85
    assert opt2.moment_2_decay == 0.95
    assert opt2.eps == 1e-7
    assert opt2.time_step == 2

    for m1, m2 in zip(opt.moment_1, opt2.moment_1):
        np.testing.assert_array_equal(m1, m2)
    for m1, m2 in zip(opt.moment_2, opt2.moment_2):
        np.testing.assert_array_equal(m1, m2)


def test_adam_file_save_load_round_trip(tmp_path):
    model = Sequential([Linear(4, 2)])
    net = Network("file_test", model, 4, 2)

    opt = ADAM(learning_rate=0.003)
    opt.parameters = net.params()

    net.zero_grad()
    out = net.forward(np.ones((1, 4)))
    net.backward(np.ones_like(out))
    opt.step()

    save_path = tmp_path / "adam_saved.npz"
    opt.save_state(save_path, net)

    assert save_path.exists()

    opt_restored = ADAM()
    opt_restored.parameters = net.params()
    opt_restored.load_state(save_path, net)

    assert opt_restored.time_step == 1
    assert opt_restored.learning_rate == 0.003
    for m1, m2 in zip(opt.moment_1, opt_restored.moment_1):
        np.testing.assert_array_equal(m1, m2)
    for m1, m2 in zip(opt.moment_2, opt_restored.moment_2):
        np.testing.assert_array_equal(m1, m2)


def test_adam_save_state_overwrite(tmp_path):
    model = Sequential([Linear(4, 2)])
    net = Network("file_test_overwrite", model, 4, 2)

    opt1 = ADAM(learning_rate=0.01)
    opt1.parameters = net.params()
    opt1.step()

    save_path = tmp_path / "adam_overwrite.npz"
    opt1.save_state(save_path, net)

    opt2 = ADAM(learning_rate=0.05)
    opt2.parameters = net.params()
    opt2.step()
    opt2.step()
    opt2.save_state(save_path, net)

    opt_loaded = ADAM()
    opt_loaded.parameters = net.params()
    opt_loaded.load_state(save_path, net)

    assert opt_loaded.learning_rate == 0.05
    assert opt_loaded.time_step == 2


def test_adam_state_dict_isolation():
    model = Sequential([Linear(2, 2)])
    net = Network("iso_test", model, 2, 2)

    opt = ADAM(learning_rate=0.001)
    opt.parameters = net.params()
    opt.step()

    state = opt.state_dict(net)
    orig_m1 = state["state"]["model.0.W"]["moment_1"].copy()

    # Mutate state dict array
    state["state"]["model.0.W"]["moment_1"] += 100.0
    # Optimizer internal moment should not change
    np.testing.assert_array_equal(opt.moment_1[0], orig_m1)

    # Mutate optimizer internal moment
    opt.moment_1[0] += 50.0
    # Next state_dict shouldn't be affected by previous export
    state2 = opt.state_dict(net)
    assert not np.array_equal(state2["state"]["model.0.W"]["moment_1"], orig_m1)


def test_adam_parameter_identity_preservation():
    model = Sequential([Linear(4, 2)])
    net = Network("ident_test", model, 4, 2)

    params_before = net.params()
    param_ids_before = [id(p) for p in params_before]

    opt = ADAM()
    opt.parameters = net.params()
    opt.step()

    state = opt.state_dict(net)

    opt_new = ADAM()
    opt_new.parameters = net.params()
    opt_new.load_state_dict(state, net)

    params_after = net.params()
    param_ids_after = [id(p) for p in params_after]

    assert param_ids_before == param_ids_after
    for p_opt, p_net in zip(opt_new.parameters, net.params()):
        assert p_opt is p_net


def test_adam_save_load_before_first_step(tmp_path):
    model = Sequential([Linear(4, 2)])
    net = Network("uninit_test", model, 4, 2)

    opt = ADAM(learning_rate=0.002)
    opt.parameters = net.params()

    state = opt.state_dict(net)
    assert state["time_step"] == 0
    assert state["state"]["model.0.W"]["moment_1"] is None
    assert state["state"]["model.0.W"]["moment_2"] is None

    save_path = tmp_path / "uninit.npz"
    opt.save_state(save_path, net)

    opt2 = ADAM()
    opt2.parameters = net.params()
    opt2.load_state(save_path, net)

    assert opt2.time_step == 0
    assert opt2.learning_rate == 0.002
    assert opt2.moment_1 == []
    assert opt2.moment_2 == []

    # Verify that opt2 can take steps normally
    net.zero_grad()
    out = net.forward(np.ones((1, 4)))
    net.backward(np.ones_like(out))
    opt2.step()

    assert opt2.time_step == 1
    assert len(opt2.moment_1) == len(net.params())


def test_adam_moments_dtype_and_backend_preservation(tmp_path):
    # Test CPU float32 and float64
    for dtype in [np.float32, np.float64]:
        model = Sequential([Linear(4, 2)])
        net = Network(f"adam_dtype_{dtype.__name__}", model, 4, 2).to(dtype=dtype)

        opt = ADAM(learning_rate=0.002)
        opt.parameters = net.params()

        net.zero_grad()
        out = net.forward(np.ones((1, 4), dtype=dtype))
        net.backward(np.ones_like(out))
        opt.step()

        save_path = tmp_path / f"adam_{dtype.__name__}.npz"
        opt.save_state(save_path, net)

        net_restored = Network(
            f"adam_target_{dtype.__name__}", Sequential([Linear(4, 2)]), 4, 2
        ).to(dtype=dtype)
        opt_restored = ADAM()
        opt_restored.parameters = net_restored.params()
        opt_restored.load_state(save_path, net_restored)

        assert opt_restored.time_step == 1
        for m1, m2 in zip(opt_restored.moment_1, opt_restored.moment_2):
            assert isinstance(m1, np.ndarray)
            assert isinstance(m2, np.ndarray)
            assert m1.dtype == dtype
            assert m2.dtype == dtype

    # Test GPU if hardware available
    if not is_gpu_available():
        return

    net_gpu = Network("adam_gpu", Sequential([Linear(4, 2)]), 4, 2).to(
        device="gpu", dtype=np.float32
    )
    opt_gpu = ADAM()
    opt_gpu.parameters = net_gpu.params()

    net_gpu.zero_grad()
    xp = get_array_module(net_gpu.params()[0].data)
    out = net_gpu.forward(xp.ones((1, 4), dtype=np.float32))
    net_gpu.backward(xp.ones_like(out))
    opt_gpu.step()

    gpu_save_path = tmp_path / "adam_gpu.npz"
    opt_gpu.save_state(gpu_save_path, net_gpu)

    # Restore onto GPU
    net_gpu_restored = Network("adam_gpu_target", Sequential([Linear(4, 2)]), 4, 2).to(
        device="gpu", dtype=np.float32
    )
    opt_gpu_restored = ADAM()
    opt_gpu_restored.parameters = net_gpu_restored.params()
    opt_gpu_restored.load_state(gpu_save_path, net_gpu_restored)

    for m1, m2 in zip(opt_gpu_restored.moment_1, opt_gpu_restored.moment_2):
        assert hasattr(m1, "get")
        assert hasattr(m2, "get")
        assert m1.dtype == np.float32
        assert m2.dtype == np.float32

    # Restore onto CPU
    net_cpu_restored = Network("adam_cpu_target", Sequential([Linear(4, 2)]), 4, 2).to(
        device="cpu", dtype=np.float32
    )
    opt_cpu_restored = ADAM()
    opt_cpu_restored.parameters = net_cpu_restored.params()
    opt_cpu_restored.load_state(gpu_save_path, net_cpu_restored)

    for m1, m2 in zip(opt_cpu_restored.moment_1, opt_cpu_restored.moment_2):
        assert isinstance(m1, np.ndarray)
        assert isinstance(m2, np.ndarray)
        assert m1.dtype == np.float32
        assert m2.dtype == np.float32
