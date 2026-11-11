# Copperhead

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

<img src="docs/images/project-logo.webp" width="50%" alt="Copperhead Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b>📋 Table of Contents</b></summary>

- [Description](#description)
- [🛠️ Prerequisites](#%EF%B8%8F-prerequisites)
- [Getting Started](#getting-started)
  - [CPU Setup (Default)](#cpu-setup-default)
  - [AMD GPU (ROCm) Acceleration Setup](#amd-gpu-rocm-acceleration-setup)
  - [NVIDIA GPU (CUDA) Setup (Alternative)](#nvidia-gpu-cuda-setup-alternative)
- [Hardware & Device Switching API](#hardware--device-switching-api)
  - [Backend Configuration](#backend-configuration)
  - [Device-Aware Models and Parameters](#device-aware-models-and-parameters)
  - [Precision: Float32 vs Float64](#precision-float32-vs-float64)
- [Usage & Training](#usage--training)
  - [MNIST Classification](#mnist-classification)
  - [Snake Reinforcement Learning (DQN)](#snake-reinforcement-learning-dqn)
- [Benchmarking](#benchmarking)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

</details>

## Description

**Copperhead** is a custom neural network framework implemented from scratch to explore the mathematical foundations of deep learning and reinforcement learning.

It features:
- **From-scratch Mathematics**: Custom implementations of linear transformations, activations (ReLU), spatial convolutions (`Conv2D`), 2D pooling (`MaxPool2D`, `AvgPool2D`), array reshaping (`Flatten`, `Concat`), and index accumulation algorithms (`im2col`, `col2im`).
- **Flexible Backend**: Runs on CPU via NumPy by default and optionally accelerates tensor mathematics on AMD GPUs via AMD ROCm and CuPy (or NVIDIA GPUs via CUDA).
- **Zero Host Roundtrips**: When running on GPU, network forward, loss calculation, backpropagation, and optimizer steps execute entirely in GPU device memory without unnecessary CPU transfers.
- **Model-Driven NN**: YAML configuration loader to instantiate and execute deep architectures declaratively.

---

## 🛠️ Prerequisites

- **Git**: [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: [Download Python](https://www.python.org/downloads/)
- **UV**: For dependency management. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- *(Optional for GPU)* **AMD ROCm**: Version 5.x or 7.x drivers installed with `rocm-smi` available.

---

## Getting Started

### CPU Setup (Default)

Copperhead works out of the box on standard CPU environments without any GPU packages:

```bash
git clone https://github.com/CogitoNTNU/Copperhead.git
cd Copperhead
uv sync
```

### AMD GPU (ROCm) Acceleration Setup

For AMD GPUs with ROCm drivers installed:

1. **Verify ROCm Driver Installation**:
   Ensure `rocm-smi` is installed and reports your AMD GPU:
   ```bash
   rocm-smi
   ```

2. **Install CuPy with ROCm Support**:
   - **For ROCm 7.x (Pre-built Wheel)**:
     ```bash
     uv sync --extra rocm
     # or using pip:
     pip install coppercore[rocm]
     ```
   - **For ROCm 5.x**:
     ```bash
     uv pip install "cupy-rocm-5-0>=13.0.0"
     ```
   - **Building from Source (Custom ROCm Version / Architecture)**:
     If using ROCm 6.x or custom hardware targets (e.g. `gfx1100`, `gfx1030`), install CuPy from source with HIP support:
     ```bash
     CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/opt/rocm HCC_AMDGPU_TARGET=gfx1100 uv pip install cupy
     ```

### NVIDIA GPU (CUDA) Setup (Alternative)

If you are using an NVIDIA GPU with CUDA 12.x drivers:
```bash
uv sync --extra cuda
# or using pip:
pip install coppercore[cuda]
```

---

## Hardware & Device Switching API

### Backend Configuration

The backend defaults to `"cpu"` (NumPy). You can query or configure the global backend dynamically:

```python
from coppercore.backend import get_backend, set_backend, is_gpu_available

print(get_backend())       # 'cpu'
print(is_gpu_available())  # True if CuPy + ROCm/CUDA runtime is active

# Explicitly switch global backend
set_backend("gpu")  # Raises clear RuntimeError with installation hints if GPU is not available
```

### Device-Aware Models and Parameters

Models, layers, and optimizers can be migrated between devices and precision modes using `.to(device, dtype)`:

```python
import numpy as np
from coppercore.model_driven_nn import Network
from coppercore.model_driven_nn.components import Sequential, Linear, ReLU
from coppercore.optimizer import ADAM

# Construct model
net = Network(
    name="mlp",
    model=Sequential([Linear(784, 128), ReLU(), Linear(128, 10)]),
    input_dim=784,
    output_dim=10,
)
optimizer = ADAM(learning_rate=1e-3)
optimizer.parameters = net.params()

# Move network and optimizer to GPU with single-precision float32
net.to(device="gpu", dtype=np.float32)
optimizer.to(device="gpu", dtype=np.float32)

# Move back to CPU with double-precision float64
net.to(device="cpu", dtype=np.float64)
optimizer.to(device="cpu", dtype=np.float64)
```

### Precision: Float32 vs Float64

- **GPU Acceleration**: Single-precision `float32` is recommended on GPU for maximum tensor throughput and reduced VRAM memory footprint.
- **CPU Default**: Double-precision `float64` is standard on CPU for maximum numerical accuracy during verification and finite-difference gradient checks.

---

## Usage & Training

### MNIST Classification

Train a YAML-defined network on the MNIST dataset:

```bash
# CPU training (float64)
uv run run-mnist --network packages/coppercore/networks/v1.yaml --epochs 5 --device cpu

# GPU accelerated training (float32)
uv run run-mnist --network packages/coppercore/networks/v1.yaml --epochs 10 --device gpu --dtype float32
```

### Snake Reinforcement Learning (DQN)

The convolutional Snake DQN agent supports both CPU (default) and AMD ROCm GPU acceleration:

- **Architecture**: Convolutional feature extractor (`Conv2D` 16 channels, `ReLU`, `Flatten`) followed by MLP decision layers (`Linear` 128 -> `ReLU` -> `Linear` 64 -> `ReLU` -> `Linear` 4).
- **CPU/GPU Partitioning**: The Snake game environment and raw transition replay buffer stay on CPU to eliminate micro-transfer latency during step simulation. Sampled mini-batches (states, actions, rewards, next states, terminations) are transferred to GPU for high-throughput batch forward and backward passes.
- **Single-State Inference**: Action selection (`act()`) receives a CPU observation from the game environment, transfers the single state to device, runs forward prediction, and returns a scalar action integer. During random exploration, neural network forward evaluation is skipped.
- **Recommended Hyperparameters**: A batch size of `128` or `256` in `float32` precision provides optimal throughput and stable gradient updates.

#### Running Snake DQN Training:

```bash
# CPU training (default)
uv run python projects/convolution/src/convolution/main.py --device cpu --dtype float32 --batch-size 128

# AMD GPU (ROCm) accelerated training
uv run python projects/convolution/src/convolution/main.py --device gpu --dtype float32 --batch-size 128 --n-steps 800000
```

#### Programmatic Usage:

```python
import numpy as np
from convolution.main import DQNAgent, make_snake_env

env = make_snake_env()

# Initialize agent on GPU with float32
agent = DQNAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    obs_shape=env.obs_shape,
    device="gpu",      # or "cpu"
    dtype=np.float32,  # or np.float64
    batch_size=128,
)
```

---

## Benchmarking

Run the built-in benchmark script to compare CPU and GPU throughput for Dense layers, Convolutions, Pooling, Adam updates, and end-to-end training batches:

```bash
uv run python benchmarks/benchmark_gpu.py --dtype float32
```

The benchmark includes stream synchronization (`cupy.cuda.Stream.synchronize`) and warmup runs to ensure accurate hardware timing measurements.

---

## Testing

Run the automated test suite across all components:

```bash
uv run pytest packages/coppercore -q
```

GPU-specific test cases automatically skip when no compatible GPU hardware or ROCm runtime is detected, allowing tests to run cleanly in standard CPU and CI environments.

---

## Troubleshooting

- **`RuntimeError: CuPy GPU backend is not available`**:
  Make sure `rocm-smi` detects your AMD GPU and install the appropriate wheel (`uv sync --extra rocm` or `uv pip install cupy-rocm-5-0`).
- **`HIP_VISIBLE_DEVICES`**:
  If multiple GPUs are present, select a specific device by setting `export HIP_VISIBLE_DEVICES=0`.
- **Conflicting CuPy packages**:
  Do not install multiple CuPy wheel variants in the same environment (e.g. `cupy-rocm-7-0` and `cupy-cuda12x` simultaneously). Use `uv sync --extra rocm` for AMD or `--extra cuda` for NVIDIA.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
