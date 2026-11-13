import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from convolution.main import DQNAgent, make_snake_env
from coppercore.backend import is_gpu_available, to_device
from coppercore.dqn.base_agent import BaseAgent
from coppercore.training.dqn_harness import evaluate

# Ensure projects/convolution is importable
root_dir = Path(__file__).resolve().parents[3]
conv_src = root_dir / "projects" / "convolution" / "src"
if str(conv_src) not in sys.path:
    sys.path.insert(0, str(conv_src))


@pytest.fixture(autouse=True)
def disable_wandb():
    """Disable wandb syncing during tests."""
    with patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
        yield


def sequential_evaluate_reference(agent, make_env, n_episodes=20, seed=10_000):
    """Reference sequential implementation of evaluate."""
    env = make_env()
    returns, scores = [], []

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        episode_return = 0.0
        done = False

        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)
        if "score" in info:
            scores.append(info["score"])

    out = {"return_mean": float(np.mean(returns))}
    if scores:
        out["score_mean"] = float(np.mean(scores))
    return out


def test_act_batch_matches_single_act_cpu():
    """Verify act_batch matches single act on CPU."""
    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
    )

    rng = np.random.default_rng(123)
    obs_batch = rng.standard_normal((10, *env.obs_shape)).astype(np.float32)

    batched_actions = agent.act_batch(obs_batch, greedy=True)
    assert isinstance(batched_actions, np.ndarray)
    assert batched_actions.shape == (10,)
    assert np.issubdtype(batched_actions.dtype, np.integer)

    single_actions = np.array(
        [agent.act(obs, greedy=True) for obs in obs_batch], dtype=np.int64
    )

    np.testing.assert_array_equal(batched_actions, single_actions)


def test_act_batch_empty_array():
    """Verify act_batch gracefully handles empty observation array."""
    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
    )
    empty_obs = np.empty((0, *env.obs_shape), dtype=np.float32)
    actions = agent.act_batch(empty_obs, greedy=True)
    assert isinstance(actions, np.ndarray)
    assert len(actions) == 0


def test_batched_evaluation_matches_sequential_cpu():
    """Verify batched evaluation exactly matches sequential evaluation on CPU."""
    env = make_snake_env()
    for n_episodes in [1, 5, 20]:
        agent = DQNAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            obs_shape=env.obs_shape,
            seed=42,
            device="cpu",
            dtype=np.float32,
        )

        seq_res = sequential_evaluate_reference(
            agent, make_snake_env, n_episodes=n_episodes, seed=10_000
        )
        batch_res = evaluate(agent, make_snake_env, n_episodes=n_episodes, seed=10_000)

        assert np.isclose(batch_res["return_mean"], seq_res["return_mean"])
        if "score_mean" in seq_res:
            assert np.isclose(batch_res["score_mean"], seq_res["score_mean"])

        assert batch_res["total_episodes"] == n_episodes
        assert batch_res["total_transitions"] > 0
        assert batch_res["duration_seconds"] >= 0.0
        assert batch_res["inference_calls"] > 0
        assert batch_res["mean_inference_batch_size"] > 0


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy/GPU is not available")
def test_act_batch_and_evaluation_gpu():
    """Verify act_batch and evaluate on CuPy GPU match CPU."""
    env = make_snake_env()
    agent_cpu = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
    )
    agent_gpu = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="gpu",
        dtype=np.float32,
    )

    # Copy weights from CPU to GPU
    for p_gpu, p_cpu in zip(agent_gpu.Q_policy.params(), agent_cpu.Q_policy.params()):
        p_gpu.data = to_device(p_cpu.data.copy(), device="gpu", dtype=np.float32)

    rng = np.random.default_rng(123)
    obs_batch = rng.standard_normal((15, *env.obs_shape)).astype(np.float32)

    actions_cpu = agent_cpu.act_batch(obs_batch, greedy=True)
    actions_gpu = agent_gpu.act_batch(obs_batch, greedy=True)
    assert isinstance(actions_gpu, np.ndarray)
    np.testing.assert_array_equal(actions_gpu, actions_cpu)

    # Evaluate across 20 episodes
    res_cpu = evaluate(agent_cpu, make_snake_env, n_episodes=20, seed=10_000)
    res_gpu = evaluate(agent_gpu, make_snake_env, n_episodes=20, seed=10_000)

    assert np.isclose(res_gpu["return_mean"], res_cpu["return_mean"])
    if "score_mean" in res_cpu:
        assert np.isclose(res_gpu["score_mean"], res_cpu["score_mean"])


class MockDeterministicEnv:
    """Mock environment that terminates after a specified number of steps."""

    def __init__(self, max_steps: int, score: int = 1):
        self.max_steps = max_steps
        self.score = score
        self.steps = 0
        self.terminated = False

    def reset(self, seed=None):
        self.steps = 0
        self.terminated = False
        return np.ones((1, 8, 8), dtype=np.float32), {"score": 0}

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Completed environment stepped again!")
        self.steps += 1
        reward = 1.0
        terminated = self.steps >= self.max_steps
        truncated = False
        self.terminated = terminated
        info = {"score": self.score if terminated else 0}
        return np.ones((1, 8, 8), dtype=np.float32), reward, terminated, truncated, info


def test_completed_environments_never_stepped_again():
    """Verify completed environments are excluded from subsequent steps."""
    step_limits = [2, 5, 1, 8, 3]
    env_pool = [
        MockDeterministicEnv(limit, score=i) for i, limit in enumerate(step_limits)
    ]

    created = 0

    def make_mock_env():
        nonlocal created
        env = env_pool[created]
        created += 1
        return env

    agent = BaseAgent(n_actions=4, seed=42)
    res = evaluate(agent, make_mock_env, n_episodes=len(step_limits), seed=100)

    assert res["total_episodes"] == 5
    assert res["total_transitions"] == sum(step_limits)
    assert np.isclose(res["return_mean"], float(np.mean(step_limits)))
    assert np.isclose(res["score_mean"], float(np.mean(range(len(step_limits)))))


def test_evaluation_zero_episodes():
    """Verify evaluation behavior with zero episodes."""
    agent = BaseAgent(n_actions=4, seed=42)
    res = evaluate(agent, make_snake_env, n_episodes=0)
    assert res["return_mean"] == 0.0
    assert res["total_episodes"] == 0
    assert res["total_transitions"] == 0
    assert res["inference_calls"] == 0


def test_agent_without_act_batch_fallback():
    """Verify evaluation works for agents that only implement act()."""

    class LegacyAgent:
        def __init__(self):
            self.rng = np.random.default_rng(42)

        def act(self, obs: np.ndarray, greedy: bool = False) -> int:
            return int(self.rng.integers(0, 4))

        def observe(self, *args, **kwargs):
            return None

    legacy_agent = LegacyAgent()
    res = evaluate(legacy_agent, make_snake_env, n_episodes=5, seed=10_000)
    assert res["total_episodes"] == 5
    assert res["total_transitions"] > 0
    assert res["inference_calls"] > 0


def test_no_training_gradients_modified_during_eval():
    """Verify evaluation does not touch or overwrite training gradients."""
    env = make_snake_env()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=42,
        device="cpu",
        dtype=np.float32,
    )

    # Set mock gradient values
    for p in agent.Q_policy.params():
        p.grad = np.ones_like(p.data) * 42.0

    _ = evaluate(agent, make_snake_env, n_episodes=5, seed=10_000)

    for p in agent.Q_policy.params():
        np.testing.assert_array_equal(p.grad, np.ones_like(p.data) * 42.0)
