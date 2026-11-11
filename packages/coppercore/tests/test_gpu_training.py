import os
from unittest.mock import patch

import numpy as np
import pytest
from coppercore.backend import is_gpu_available
from coppercore.buffer import ReplayBuffer
from coppercore.model_driven_nn import Network
from coppercore.model_driven_nn.components import Linear, ReLU, Sequential
from coppercore.optimizer import ADAM, SGD
from coppercore.training.dqn_train import dqn_update
from coppercore.training.mnist_train import train as train_mnist
from coppercore.training.run_mnist import evaluate, parse_args


@pytest.fixture(autouse=True)
def disable_wandb():
    """Disable wandb syncing during tests."""
    with patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
        yield


def test_mnist_training_cpu_float32():
    """Verify MNIST training executes on CPU in float32 precision."""
    net = Network(
        name="test_mnist_net",
        model=Sequential([Linear(784, 32), ReLU(), Linear(32, 10)]),
        input_dim=784,
        output_dim=10,
    )
    opt = SGD(learning_rate=0.01)

    trained_net = train_mnist(
        net,
        opt,
        epochs=1,
        batch_size=128,
        seed=42,
        device="cpu",
        dtype=np.float32,
    )

    for p in trained_net.params():
        assert p.data.dtype == np.float32
        assert p.grad.dtype == np.float32

    loss, acc = evaluate(trained_net, seed=42, device="cpu", dtype=np.float32)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_dqn_update_cpu_float32():
    """Verify Snake DQN update runs on CPU in float32."""
    policy = Network(
        name="policy",
        model=Sequential([Linear(10, 16), ReLU(), Linear(16, 4)]),
        input_dim=10,
        output_dim=4,
    )
    target = Network(
        name="target",
        model=Sequential([Linear(10, 16), ReLU(), Linear(16, 4)]),
        input_dim=10,
        output_dim=4,
    )
    policy.to(dtype=np.float32)
    target.to(dtype=np.float32)

    opt = ADAM(learning_rate=1e-3)
    opt.to(dtype=np.float32)
    opt.parameters = policy.params()

    buf = ReplayBuffer(capacity=100, obs_shape=(10,), obs_dtype=np.float32)
    rng = np.random.default_rng(42)

    for _ in range(32):
        s = rng.standard_normal(10).astype(np.float32)
        a = int(rng.integers(0, 4))
        r = float(rng.standard_normal())
        s_next = rng.standard_normal(10).astype(np.float32)
        term = bool(rng.integers(0, 2))
        buf.add(s, a, r, s_next, term)

    loss = dqn_update(
        policy,
        target,
        opt,
        buf,
        batch_size=16,
        gamma=0.99,
        rng=rng,
        device="cpu",
        dtype=np.float32,
    )
    assert isinstance(loss, (float, np.floating))
    assert float(loss) >= 0.0
    for p in policy.params():
        assert p.data.dtype == np.float32


def test_run_mnist_cli_args():
    """Verify parse_args supports --device and --dtype flags."""
    with patch("sys.argv", ["run_mnist.py", "--device", "gpu", "--dtype", "float32"]):
        args = parse_args()
        assert args.device == "gpu"
        assert args.dtype == "float32"


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy/GPU is not available")
def test_real_gpu_mnist_and_dqn():
    """Verify MNIST training and DQN update on real GPU hardware."""
    import cupy as cp

    # 1. MNIST GPU training
    net = Network(
        name="gpu_mnist_net",
        model=Sequential([Linear(784, 32), ReLU(), Linear(32, 10)]),
        input_dim=784,
        output_dim=10,
    )
    opt = SGD(learning_rate=0.01)

    trained_net = train_mnist(
        net,
        opt,
        epochs=1,
        batch_size=128,
        seed=42,
        device="gpu",
        dtype=np.float32,
    )

    for p in trained_net.params():
        assert isinstance(p.data, cp.ndarray)
        assert p.data.dtype == np.float32

    # 2. DQN GPU update
    policy = Network(
        name="gpu_policy",
        model=Sequential([Linear(10, 16), ReLU(), Linear(16, 4)]),
        input_dim=10,
        output_dim=4,
    )
    target = Network(
        name="gpu_target",
        model=Sequential([Linear(10, 16), ReLU(), Linear(16, 4)]),
        input_dim=10,
        output_dim=4,
    )
    policy.to(device="gpu", dtype=np.float32)
    target.to(device="gpu", dtype=np.float32)

    opt_dqn = ADAM(learning_rate=1e-3)
    opt_dqn.to(device="gpu", dtype=np.float32)
    opt_dqn.parameters = policy.params()

    buf = ReplayBuffer(capacity=100, obs_shape=(10,), obs_dtype=np.float32)
    rng = np.random.default_rng(42)

    for _ in range(32):
        s = rng.standard_normal(10).astype(np.float32)
        a = int(rng.integers(0, 4))
        r = float(rng.standard_normal())
        s_next = rng.standard_normal(10).astype(np.float32)
        term = bool(rng.integers(0, 2))
        buf.add(s, a, r, s_next, term)

    loss = dqn_update(
        policy,
        target,
        opt_dqn,
        buf,
        batch_size=16,
        gamma=0.99,
        rng=rng,
        device="gpu",
        dtype=np.float32,
    )
    assert isinstance(loss, (float, np.floating))
    assert float(loss) >= 0.0
    for p in policy.params():
        assert isinstance(p.data, cp.ndarray)


def test_snake_dqn_agent_cpu_and_inference():
    """Verify convolutional Snake DQNAgent behavior on CPU."""
    import sys
    from pathlib import Path

    # Add convolution project to sys.path
    conv_src = Path(__file__).resolve().parents[3] / "projects" / "convolution" / "src"
    if str(conv_src) not in sys.path:
        sys.path.insert(0, str(conv_src))

    from convolution.main import DQNAgent, make_snake_env

    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
        n_train_start=10,
        n_target_start=5,
        train_freq=2,
        batch_size=4,
    )

    # 1. Parameter shapes and device/dtype
    assert "snake-dqn-policy" in agent.Q_policy.name
    assert "snake-dqn-target" in agent.Q_target.name
    custom_agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        name="custom-dqn",
    )
    assert "custom-dqn-policy" in custom_agent.Q_policy.name
    assert "custom-dqn-target" in custom_agent.Q_target.name

    for p in agent.Q_policy.params():
        assert p.data.dtype == np.float32
        assert isinstance(p.data, np.ndarray)
    for p in agent.Q_target.params():
        assert p.data.dtype == np.float32
        assert isinstance(p.data, np.ndarray)

    # 2. params serialization property
    params_dict = agent.params
    assert len(params_dict) > 0
    for val in params_dict.values():
        assert isinstance(val, np.ndarray)

    # 3. act() greedy and random
    obs, _ = env.reset(seed=42)
    greedy_action = agent.act(obs, greedy=True)
    assert isinstance(greedy_action, int)
    assert 0 <= greedy_action < env.n_actions

    random_action = agent.act(obs, greedy=False)
    assert isinstance(random_action, int)
    assert 0 <= random_action < env.n_actions

    # 4. observe() transitions and updates
    for _ in range(12):
        action = agent.act(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        metrics = agent.observe(obs, action, reward, next_obs, term)
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs

    # After 12 steps (> n_train_start=10, train_freq=2), metrics should be returned
    assert metrics is not None
    assert "loss" in metrics
    assert "epsilon" in metrics
    assert isinstance(metrics["loss"], float)

    # 5. Missing GPU error
    if not is_gpu_available():
        with pytest.raises(RuntimeError):
            DQNAgent(
                n_states=env.n_states,
                n_actions=env.n_actions,
                obs_shape=env.obs_shape,
                device="gpu",
            )
