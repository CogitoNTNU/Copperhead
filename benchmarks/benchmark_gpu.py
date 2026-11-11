"""Benchmark script comparing NumPy CPU and CuPy GPU performance in Copperhead."""

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

from coppercore.backend import is_gpu_available, to_device
from coppercore.buffer import ReplayBuffer
from coppercore.model_driven_nn import Network
from coppercore.model_driven_nn.components import (
    Conv2D,
    Flatten,
    Linear,
    MaxPool2D,
    ReLU,
    Sequential,
)
from coppercore.optimizer import ADAM, SGD
from coppercore.training.dqn_train import dqn_update
from coppercore.training.mnist_train import softmax_cross_entropy_loss


def sync_device(device: str) -> None:
    """Synchronize the active device stream to ensure accurate timing."""
    if device == "gpu":
        try:
            import cupy as cp

            cp.cuda.get_current_stream().synchronize()
        except (ImportError, AttributeError):
            pass


def benchmark_linear(
    batch_size: int = 128,
    in_dim: int = 1024,
    out_dim: int = 1024,
    device: str = "cpu",
    dtype: Any = np.float32,
    warmup: int = 5,
    iterations: int = 20,
) -> float:
    fc = Linear(in_dim, out_dim)
    fc.to(device=device, dtype=dtype)

    x = np.random.randn(batch_size, in_dim).astype(dtype)
    x = to_device(x, device=device, dtype=dtype)
    d_out = np.ones((batch_size, out_dim), dtype=dtype)
    d_out = to_device(d_out, device=device, dtype=dtype)

    for _ in range(warmup):
        fc.forward(x, cache=True)
        fc.backward(d_out)
    sync_device(device)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fc.forward(x, cache=True)
        fc.backward(d_out)
        sync_device(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_conv2d(
    batch_size: int = 32,
    in_channels: int = 16,
    out_channels: int = 32,
    height: int = 28,
    width: int = 28,
    device: str = "cpu",
    dtype: Any = np.float32,
    warmup: int = 3,
    iterations: int = 10,
) -> float:
    conv = Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    )
    conv.to(device=device, dtype=dtype)

    x = np.random.randn(batch_size, in_channels, height, width).astype(dtype)
    x = to_device(x, device=device, dtype=dtype)
    out_shape = (batch_size, out_channels, height, width)
    d_out = np.ones(out_shape, dtype=dtype)
    d_out = to_device(d_out, device=device, dtype=dtype)

    for _ in range(warmup):
        conv.forward(x, cache=True)
        conv.backward(d_out)
    sync_device(device)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        conv.forward(x, cache=True)
        conv.backward(d_out)
        sync_device(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_maxpool2d(
    batch_size: int = 32,
    channels: int = 32,
    height: int = 28,
    width: int = 28,
    device: str = "cpu",
    dtype: Any = np.float32,
    warmup: int = 3,
    iterations: int = 10,
) -> float:
    pool = MaxPool2D(kernel_size=2, stride=2)
    pool.to(device=device, dtype=dtype)

    x = np.random.randn(batch_size, channels, height, width).astype(dtype)
    x = to_device(x, device=device, dtype=dtype)
    out_shape = (batch_size, channels, height // 2, width // 2)
    d_out = np.ones(out_shape, dtype=dtype)
    d_out = to_device(d_out, device=device, dtype=dtype)

    for _ in range(warmup):
        pool.forward(x, cache=True)
        pool.backward(d_out)
    sync_device(device)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        pool.forward(x, cache=True)
        pool.backward(d_out)
        sync_device(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_adam(
    param_size: int = 500_000,
    device: str = "cpu",
    dtype: Any = np.float32,
    warmup: int = 5,
    iterations: int = 20,
) -> float:
    p_data = np.random.randn(param_size).astype(dtype)
    p_grad = np.random.randn(param_size).astype(dtype)

    fc = Linear(10, 10)
    fc.W.data = to_device(p_data, device=device, dtype=dtype)
    fc.W.grad = to_device(p_grad, device=device, dtype=dtype)

    opt = ADAM(learning_rate=0.001)
    opt.to(device=device, dtype=dtype)
    opt.parameters = [fc.W]

    for _ in range(warmup):
        opt.step()
    sync_device(device)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        opt.step()
        sync_device(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_snake_dqn_inference(
    device: str = "cpu",
    dtype: Any = np.float32,
    warmup: int = 10,
    iterations: int = 50,
) -> float:
    """Benchmark single observation action inference latency."""
    total_dim = 64
    conv_out_channels = 16
    conv_flat_dim = conv_out_channels * 8 * 8

    net = Network(
        "snake-dqn",
        Sequential(
            [
                Conv2D(1, conv_out_channels, kernel_size=3, padding=1, input_shape=(1, 8, 8)),
                ReLU(),
                Flatten(),
                Linear(conv_flat_dim, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 4),
            ]
        ),
        input_dim=total_dim,
        output_dim=4,
    )
    net.to(device=device, dtype=dtype)

    obs = np.random.randn(1, 8, 8).astype(dtype)
    for _ in range(warmup):
        obs_dev = to_device(obs[None, ...], device=device, dtype=dtype)
        q = net.predict(obs_dev)
        _ = int(np.asarray(q.argmax(axis=1))[0])
    sync_device(device)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        obs_dev = to_device(obs[None, ...], device=device, dtype=dtype)
        q = net.predict(obs_dev)
        _ = int(np.asarray(q.argmax(axis=1))[0])
        sync_device(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_snake_dqn_update(
    batch_size: int = 128,
    device: str = "cpu",
    dtype: Any = np.float32,
    warmup: int = 3,
    iterations: int = 15,
) -> float:
    """Benchmark full convolutional Snake DQN batch update latency."""
    obs_shape = (1, 8, 8)
    conv_out_channels = 16
    conv_flat_dim = conv_out_channels * 8 * 8
    total_dim = 64

    policy = Network(
        "snake-dqn-policy",
        Sequential(
            [
                Conv2D(1, conv_out_channels, kernel_size=3, padding=1, input_shape=(1, 8, 8)),
                ReLU(),
                Flatten(),
                Linear(conv_flat_dim, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 4),
            ]
        ),
        input_dim=total_dim,
        output_dim=4,
    )
    target = Network(
        "snake-dqn-target",
        Sequential(
            [
                Conv2D(1, conv_out_channels, kernel_size=3, padding=1, input_shape=(1, 8, 8)),
                ReLU(),
                Flatten(),
                Linear(conv_flat_dim, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 4),
            ]
        ),
        input_dim=total_dim,
        output_dim=4,
    )
    policy.to(device=device, dtype=dtype)
    target.to(device=device, dtype=dtype)

    opt = ADAM(learning_rate=5e-4)
    opt.parameters = policy.params()
    opt.to(device=device, dtype=dtype)

    buf = ReplayBuffer(capacity=batch_size * 2, obs_shape=obs_shape, obs_dtype=np.float32)
    rng = np.random.default_rng(42)

    for _ in range(batch_size * 2):
        s = rng.standard_normal(obs_shape).astype(np.float32)
        a = int(rng.integers(0, 4))
        r = float(rng.standard_normal())
        s_next = rng.standard_normal(obs_shape).astype(np.float32)
        term = bool(rng.integers(0, 2))
        buf.add(s, a, r, s_next, term)

    for _ in range(warmup):
        dqn_update(policy, target, opt, buf, batch_size, 0.99, rng, device=device, dtype=dtype)
    sync_device(device)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        dqn_update(policy, target, opt, buf, batch_size, 0.99, rng, device=device, dtype=dtype)
        sync_device(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_train_step(
    batch_size: int = 128,
    device: str = "cpu",
    dtype: Any = np.float32,
    warmup: int = 3,
    iterations: int = 15,
) -> float:
    net = Network(
        name="bench_mlp",
        model=Sequential([Linear(784, 128), ReLU(), Linear(128, 10)]),
        input_dim=784,
        output_dim=10,
    )
    net.to(device=device, dtype=dtype)

    opt = SGD(learning_rate=0.01)
    opt.to(device=device, dtype=dtype)
    opt.parameters = net.params()

    x = np.random.randn(batch_size, 784).astype(dtype)
    x = to_device(x, device=device, dtype=dtype)
    y = np.random.randint(0, 10, size=batch_size).astype(np.int64)
    y = to_device(y, device=device)

    for _ in range(warmup):
        net.zero_grad()
        logits = net.forward(x)
        _, grad_loss = softmax_cross_entropy_loss(logits, y)
        net.backward(grad_loss)
        opt.step()
    sync_device(device)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        net.zero_grad()
        logits = net.forward(x)
        _, grad_loss = softmax_cross_entropy_loss(logits, y)
        net.backward(grad_loss)
        opt.step()
        sync_device(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def run_benchmarks(dtype: Any = np.float32) -> None:
    gpu_available = is_gpu_available()
    dtype_name = "float32" if dtype == np.float32 else "float64"

    print("=" * 80)
    print(f" Copperhead Benchmark Suite (Precision: {dtype_name})")
    print(f" GPU (ROCm/CuPy) Available: {gpu_available}")
    print("=" * 80)

    benchmarks = [
        ("Linear Fwd+Bwd (1024->1024, B=128)", lambda d, t: benchmark_linear(device=d, dtype=t)),
        ("Conv2D Fwd+Bwd (16->32, 28x28, B=32)", lambda d, t: benchmark_conv2d(device=d, dtype=t)),
        ("MaxPool2D Fwd+Bwd (32ch, 28x28, B=32)", lambda d, t: benchmark_maxpool2d(device=d, dtype=t)),
        ("ADAM Optimizer Step (500k params)", lambda d, t: benchmark_adam(device=d, dtype=t)),
        ("MLP End-to-End Train Step (B=128)", lambda d, t: benchmark_train_step(device=d, dtype=t)),
        ("Snake DQN 1-State Inference (1x8x8)", lambda d, t: benchmark_snake_dqn_inference(device=d, dtype=t)),
        ("Snake DQN Update (B=64)", lambda d, t: benchmark_snake_dqn_update(batch_size=64, device=d, dtype=t)),
        ("Snake DQN Update (B=128)", lambda d, t: benchmark_snake_dqn_update(batch_size=128, device=d, dtype=t)),
        ("Snake DQN Update (B=256)", lambda d, t: benchmark_snake_dqn_update(batch_size=256, device=d, dtype=t)),
        ("Snake DQN Update (B=512)", lambda d, t: benchmark_snake_dqn_update(batch_size=512, device=d, dtype=t)),
    ]

    header = f"{'Benchmark Task':<48} | {'CPU (ms)':<10} | {'GPU (ms)':<10} | {'Speedup':<8}"
    print(header)
    print("-" * len(header))

    for name, bench_fn in benchmarks:
        cpu_ms = bench_fn("cpu", dtype)
        if gpu_available:
            gpu_ms = bench_fn("gpu", dtype)
            speedup = f"{cpu_ms / gpu_ms:.2f}x"
            gpu_str = f"{gpu_ms:.2f}"
        else:
            gpu_str = "N/A"
            speedup = "N/A"

        print(f"{name:<48} | {cpu_ms:<10.2f} | {gpu_str:<10} | {speedup:<8}")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU performance in Copperhead.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision for benchmark.",
    )
    args = parser.parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64
    run_benchmarks(dtype=dtype)


if __name__ == "__main__":
    main()
