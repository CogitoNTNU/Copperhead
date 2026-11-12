"""Benchmark script comparing sequential and batched DQN evaluation on CPU and GPU."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure projects are importable
root_dir = Path(__file__).resolve().parents[1]
conv_src = root_dir / "projects" / "convolution" / "src"
if str(conv_src) not in sys.path:
    sys.path.insert(0, str(conv_src))

from convolution.main import DQNAgent, make_snake_env
from coppercore.backend import is_gpu_available, to_device
from coppercore.training.dqn_harness import evaluate


def sync_device(device: str) -> None:
    """Synchronize device stream if GPU is active."""
    if device == "gpu":
        try:
            import cupy as cp

            cp.cuda.get_current_stream().synchronize()
        except (ImportError, AttributeError):
            pass


def sequential_evaluate(agent, make_env, n_episodes: int = 20, seed: int = 10_000) -> dict:
    """Sequential baseline evaluation harness."""
    start_time = time.perf_counter()
    env = make_env()
    returns, scores = [], []
    total_transitions = 0
    inference_calls = 0
    total_inference_time = 0.0

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        episode_return = 0.0
        done = False

        while not done:
            t0 = time.perf_counter()
            action = agent.act(obs, greedy=True)
            sync_device(getattr(agent, "device", "cpu"))
            total_inference_time += time.perf_counter() - t0
            inference_calls += 1

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            total_transitions += 1
            done = terminated or truncated

        returns.append(episode_return)
        if "score" in info:
            scores.append(info["score"])

    if hasattr(env, "close") and callable(env.close):
        env.close()

    total_duration = time.perf_counter() - start_time

    out = {
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "duration_seconds": float(total_duration),
        "total_episodes": n_episodes,
        "total_transitions": total_transitions,
        "env_steps_per_second": (
            float(total_transitions / total_duration) if total_duration > 0 else 0.0
        ),
        "inference_calls": inference_calls,
        "mean_inference_batch_size": 1.0,
        "mean_inference_latency_seconds": (
            float(total_inference_time / inference_calls) if inference_calls > 0 else 0.0
        ),
        "inference_duration_seconds": float(total_inference_time),
    }
    if scores:
        out["score_mean"] = float(np.mean(scores))
    return out


def run_evaluation_benchmark(
    episode_counts=(1, 5, 10, 20, 32),
    dtype: Any = np.float32,
    seed: int = 10_000,
) -> None:
    gpu_available = is_gpu_available()
    dtype_name = "float32" if dtype == np.float32 else "float64"

    print("=" * 105)
    print(f" Copperhead DQN Evaluation Benchmark (Precision: {dtype_name})")
    print(f" GPU (ROCm/CuPy) Available: {gpu_available}")
    print("=" * 105)

    env_proto = make_snake_env()
    obs_shape = env_proto.obs_shape
    n_states = env_proto.n_states
    n_actions = env_proto.n_actions

    # Create CPU and GPU agents with identical weights
    agent_cpu = DQNAgent(
        n_states=n_states,
        n_actions=n_actions,
        obs_shape=obs_shape,
        seed=42,
        device="cpu",
        dtype=dtype,
    )

    if gpu_available:
        agent_gpu = DQNAgent(
            n_states=n_states,
            n_actions=n_actions,
            obs_shape=obs_shape,
            seed=42,
            device="gpu",
            dtype=dtype,
        )
        for p_gpu, p_cpu in zip(agent_gpu.Q_policy.params(), agent_cpu.Q_policy.params()):
            p_gpu.data = to_device(p_cpu.data.copy(), device="gpu", dtype=dtype)
    else:
        agent_gpu = None

    # Warmup runs to avoid JIT / cold-start artifacts
    print("Performing warmup runs...")
    _ = sequential_evaluate(agent_cpu, make_snake_env, n_episodes=2, seed=seed)
    _ = evaluate(agent_cpu, make_snake_env, n_episodes=2, seed=seed)
    if gpu_available:
        _ = sequential_evaluate(agent_gpu, make_snake_env, n_episodes=2, seed=seed)
        _ = evaluate(agent_gpu, make_snake_env, n_episodes=2, seed=seed)
    print("Warmup complete.\n")

    # Benchmarking table header
    header = (
        f"{'Episodes':<10} | {'Mode':<18} | {'Duration (s)':<14} | {'Steps':<8} "
        f"| {'Avg Batch':<10} | {'Steps/s':<12} | {'Inf Time (s)':<14} | {'Speedup':<10}"
    )
    print(header)
    print("-" * len(header))

    for n_episodes in episode_counts:
        # Sequential CPU
        res_seq_cpu = sequential_evaluate(
            agent_cpu, make_snake_env, n_episodes=n_episodes, seed=seed
        )
        print(
            f"{n_episodes:<10} | {'Seq CPU':<18} | {res_seq_cpu['duration_seconds']:<14.4f} "
            f"| {res_seq_cpu['total_transitions']:<8} | {res_seq_cpu['mean_inference_batch_size']:<10.1f} "
            f"| {res_seq_cpu['env_steps_per_second']:<12.1f} | {res_seq_cpu['inference_duration_seconds']:<14.4f} "
            f"| {'1.00x (base)':<10}"
        )

        # Batched CPU
        res_bat_cpu = evaluate(
            agent_cpu, make_snake_env, n_episodes=n_episodes, seed=seed
        )
        speedup_cpu = res_seq_cpu["duration_seconds"] / max(res_bat_cpu["duration_seconds"], 1e-9)
        print(
            f"{n_episodes:<10} | {'Batched CPU':<18} | {res_bat_cpu['duration_seconds']:<14.4f} "
            f"| {res_bat_cpu['total_transitions']:<8} | {res_bat_cpu['mean_inference_batch_size']:<10.1f} "
            f"| {res_bat_cpu['env_steps_per_second']:<12.1f} | {res_bat_cpu['inference_duration_seconds']:<14.4f} "
            f"| {speedup_cpu:<9.2f}x"
        )

        if gpu_available:
            # Sequential GPU
            res_seq_gpu = sequential_evaluate(
                agent_gpu, make_snake_env, n_episodes=n_episodes, seed=seed
            )
            speedup_seq_gpu = res_seq_cpu["duration_seconds"] / max(res_seq_gpu["duration_seconds"], 1e-9)
            print(
                f"{n_episodes:<10} | {'Seq GPU':<18} | {res_seq_gpu['duration_seconds']:<14.4f} "
                f"| {res_seq_gpu['total_transitions']:<8} | {res_seq_gpu['mean_inference_batch_size']:<10.1f} "
                f"| {res_seq_gpu['env_steps_per_second']:<12.1f} | {res_seq_gpu['inference_duration_seconds']:<14.4f} "
                f"| {speedup_seq_gpu:<9.2f}x"
            )

            # Batched GPU
            res_bat_gpu = evaluate(
                agent_gpu, make_snake_env, n_episodes=n_episodes, seed=seed
            )
            speedup_bat_gpu = res_seq_gpu["duration_seconds"] / max(res_bat_gpu["duration_seconds"], 1e-9)
            speedup_vs_seq_cpu = res_seq_cpu["duration_seconds"] / max(res_bat_gpu["duration_seconds"], 1e-9)
            print(
                f"{n_episodes:<10} | {'Batched GPU':<18} | {res_bat_gpu['duration_seconds']:<14.4f} "
                f"| {res_bat_gpu['total_transitions']:<8} | {res_bat_gpu['mean_inference_batch_size']:<10.1f} "
                f"| {res_bat_gpu['env_steps_per_second']:<12.1f} | {res_bat_gpu['inference_duration_seconds']:<14.4f} "
                f"| {speedup_bat_gpu:<9.2f}x (GPU) / {speedup_vs_seq_cpu:.2f}x (all)"
            )

        print("-" * len(header))

    print("=" * 105)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sequential vs. batched DQN evaluation."
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision for benchmark.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 32],
        help="List of episode counts to evaluate.",
    )
    args = parser.parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64
    run_evaluation_benchmark(episode_counts=args.episodes, dtype=dtype)


if __name__ == "__main__":
    main()
