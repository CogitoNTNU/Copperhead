import numpy as np
from coppercore.model_driven_nn.components import Linear, ReLU, Sequential
from coppercore.model_driven_nn.network import Network
from coppercore.optimizer.adam import ADAM
from coppercore.optimizer.sgd import SGD


def create_deterministic_network() -> Network:
    np.random.seed(42)
    model = Sequential([Linear(8, 6), ReLU(), Linear(6, 4), ReLU(), Linear(4, 2)])
    return Network("test_net", model, 8, 2)


def test_adam_interrupted_vs_uninterrupted_training(tmp_path):
    # Setup fixed deterministic inputs and pseudo-targets
    np.random.seed(123)
    num_steps = 100
    interrupt_step = 40
    batch_size = 4
    input_dim = 8
    output_dim = 2

    inputs = [np.random.randn(batch_size, input_dim) for _ in range(num_steps)]
    targets = [np.random.randn(batch_size, output_dim) for _ in range(num_steps)]

    # --- Run A: Uninterrupted ---
    net_a = create_deterministic_network()
    opt_a = ADAM(
        learning_rate=0.005, moment_1_decay=0.9, moment_2_decay=0.999, eps=1e-8
    )
    opt_a.parameters = net_a.params()

    for step in range(num_steps):
        net_a.zero_grad()
        out = net_a.forward(inputs[step])
        # Simple MSE gradient: d_loss/d_out = 2 * (out - target) / batch_size
        grad = 2.0 * (out - targets[step]) / batch_size
        net_a.backward(grad)
        opt_a.step()

    final_weights_a = [p.data.copy() for p in net_a.params()]
    final_m1_a = [m.copy() for m in opt_a.moment_1]
    final_m2_a = [m.copy() for m in opt_a.moment_2]
    final_timestep_a = opt_a.time_step

    # --- Run B: Interrupted and Resumed ---
    net_b = create_deterministic_network()
    opt_b = ADAM(
        learning_rate=0.005, moment_1_decay=0.9, moment_2_decay=0.999, eps=1e-8
    )
    opt_b.parameters = net_b.params()

    # Step 0 to interrupt_step
    for step in range(interrupt_step):
        net_b.zero_grad()
        out = net_b.forward(inputs[step])
        grad = 2.0 * (out - targets[step]) / batch_size
        net_b.backward(grad)
        opt_b.step()

    # Save network weights and optimizer checkpoint
    net_ckpt_file = tmp_path / "weights.npz"
    opt_ckpt_file = tmp_path / "opt.npz"

    net_b.save_weights(net_ckpt_file)
    opt_b.save_state(opt_ckpt_file, net_b)

    # Reconstruct new network and new optimizer from files
    net_resumed = create_deterministic_network()
    net_resumed.load_weights(net_ckpt_file)

    opt_resumed = ADAM()
    opt_resumed.parameters = net_resumed.params()
    opt_resumed.load_state(opt_ckpt_file, net_resumed)

    assert opt_resumed.time_step == interrupt_step
    assert opt_resumed.learning_rate == opt_b.learning_rate
    assert opt_resumed.moment_1_decay == opt_b.moment_1_decay
    assert opt_resumed.moment_2_decay == opt_b.moment_2_decay
    assert opt_resumed.eps == opt_b.eps

    # Verify moments before resuming match opt_b
    for m_res, m_orig in zip(opt_resumed.moment_1, opt_b.moment_1):
        np.testing.assert_array_equal(m_res, m_orig)
    for m_res, m_orig in zip(opt_resumed.moment_2, opt_b.moment_2):
        np.testing.assert_array_equal(m_res, m_orig)

    # Continue training from interrupt_step to num_steps
    for step in range(interrupt_step, num_steps):
        net_resumed.zero_grad()
        out = net_resumed.forward(inputs[step])
        grad = 2.0 * (out - targets[step]) / batch_size
        net_resumed.backward(grad)
        opt_resumed.step()

    # --- Compare Run A vs Run B ---
    assert opt_resumed.time_step == final_timestep_a

    for p_a, p_b in zip(final_weights_a, [p.data for p in net_resumed.params()]):
        np.testing.assert_allclose(p_a, p_b, rtol=1e-12, atol=1e-12)

    for m1_a, m1_b in zip(final_m1_a, opt_resumed.moment_1):
        np.testing.assert_allclose(m1_a, m1_b, rtol=1e-12, atol=1e-12)

    for m2_a, m2_b in zip(final_m2_a, opt_resumed.moment_2):
        np.testing.assert_allclose(m2_a, m2_b, rtol=1e-12, atol=1e-12)


def test_sgd_interrupted_vs_uninterrupted_training(tmp_path):
    np.random.seed(456)
    num_steps = 50
    interrupt_step = 20
    batch_size = 4
    input_dim = 8
    output_dim = 2

    inputs = [np.random.randn(batch_size, input_dim) for _ in range(num_steps)]
    targets = [np.random.randn(batch_size, output_dim) for _ in range(num_steps)]

    # --- Run A: Uninterrupted ---
    net_a = create_deterministic_network()
    opt_a = SGD(learning_rate=0.02)
    opt_a.parameters = net_a.params()

    for step in range(num_steps):
        net_a.zero_grad()
        out = net_a.forward(inputs[step])
        grad = 2.0 * (out - targets[step]) / batch_size
        net_a.backward(grad)
        opt_a.step()

    final_weights_a = [p.data.copy() for p in net_a.params()]

    # --- Run B: Interrupted and Resumed ---
    net_b = create_deterministic_network()
    opt_b = SGD(learning_rate=0.02)
    opt_b.parameters = net_b.params()

    for step in range(interrupt_step):
        net_b.zero_grad()
        out = net_b.forward(inputs[step])
        grad = 2.0 * (out - targets[step]) / batch_size
        net_b.backward(grad)
        opt_b.step()

    net_ckpt_file = tmp_path / "sgd_weights.npz"
    opt_ckpt_file = tmp_path / "sgd_opt.npz"

    net_b.save_weights(net_ckpt_file)
    opt_b.save_state(opt_ckpt_file, net_b)

    net_resumed = create_deterministic_network()
    net_resumed.load_weights(net_ckpt_file)

    opt_resumed = SGD(learning_rate=0.001)
    opt_resumed.parameters = net_resumed.params()
    opt_resumed.load_state(opt_ckpt_file, net_resumed)

    assert opt_resumed.learning_rate == 0.02

    for step in range(interrupt_step, num_steps):
        net_resumed.zero_grad()
        out = net_resumed.forward(inputs[step])
        grad = 2.0 * (out - targets[step]) / batch_size
        net_resumed.backward(grad)
        opt_resumed.step()

    for p_a, p_b in zip(final_weights_a, [p.data for p in net_resumed.params()]):
        np.testing.assert_allclose(p_a, p_b, rtol=1e-12, atol=1e-12)


def test_adam_resume_before_first_step(tmp_path):
    np.random.seed(789)
    num_steps = 30
    batch_size = 4
    input_dim = 8
    output_dim = 2

    inputs = [np.random.randn(batch_size, input_dim) for _ in range(num_steps)]
    targets = [np.random.randn(batch_size, output_dim) for _ in range(num_steps)]

    net_a = create_deterministic_network()
    opt_a = ADAM(learning_rate=0.003)
    opt_a.parameters = net_a.params()

    # Save state before any step is taken
    opt_ckpt_file = tmp_path / "adam_uninit.npz"
    opt_a.save_state(opt_ckpt_file, net_a)

    # Load into fresh network and optimizer
    net_b = create_deterministic_network()
    opt_b = ADAM()
    opt_b.parameters = net_b.params()
    opt_b.load_state(opt_ckpt_file, net_b)

    assert opt_b.time_step == 0
    assert opt_b.learning_rate == 0.003

    # Run training for both
    for step in range(num_steps):
        net_a.zero_grad()
        out_a = net_a.forward(inputs[step])
        grad_a = 2.0 * (out_a - targets[step]) / batch_size
        net_a.backward(grad_a)
        opt_a.step()

        net_b.zero_grad()
        out_b = net_b.forward(inputs[step])
        grad_b = 2.0 * (out_b - targets[step]) / batch_size
        net_b.backward(grad_b)
        opt_b.step()

    for p_a, p_b in zip(net_a.params(), net_b.params()):
        np.testing.assert_allclose(p_a.data, p_b.data, rtol=1e-12, atol=1e-12)

    for m1_a, m1_b in zip(opt_a.moment_1, opt_b.moment_1):
        np.testing.assert_allclose(m1_a, m1_b, rtol=1e-12, atol=1e-12)

    for m2_a, m2_b in zip(opt_a.moment_2, opt_b.moment_2):
        np.testing.assert_allclose(m2_a, m2_b, rtol=1e-12, atol=1e-12)
