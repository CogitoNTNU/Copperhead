import argparse
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from coppercore.backend import get_array_module, is_gpu_available, to_device, to_numpy
from coppercore.buffer import ReplayBuffer
from coppercore.dqn.base_agent import BaseAgent
from coppercore.dqn.snake_env import Event, SnakeEnv
from coppercore.dqn.snake_render import RenderedSnakeEnv
from coppercore.model_driven_nn.components import (
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    Sequential,
)
from coppercore.model_driven_nn.network import Network
from coppercore.optimizer import ADAM
from coppercore.training.dqn_harness import train
from coppercore.training.dqn_train import dqn_update

import wandb

REWARDS = {Event.MOVED: -0.0001, Event.ATE: 1.0, Event.DIED: -10.0, Event.WON: 100.0}

WIDTH = 20
HEIGHT = 20


def make_snake_env():
    return SnakeEnv(
        width=WIDTH,
        height=HEIGHT,
        seed=0,
        max_steps_no_food=WIDTH * HEIGHT * 2,
        rewards=REWARDS,
        obs_mode="board",
        obs_window_radius=2,
        movement_rule=True,
    )


class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        obs_dim: int | None = None,
        obs_shape: tuple[int, ...] | None = None,
        seed: int = 0,
        device: str = "cpu",
        dtype: Any = np.float32,
        n_train_start: int = 20_000,
        n_target_start: int = 10_000,
        train_freq: int = 4,
        batch_size: int = 128,
        replay_capacity: int = 200_000,
        name: str = "snake-dqn",
    ):
        self.name = name
        self.n_state = n_states
        self.n_actions = n_actions
        self.device = device.strip().lower() if isinstance(device, str) else "cpu"
        self.dtype = dtype

        if self.device == "gpu" and not is_gpu_available():
            raise RuntimeError(
                "Requested device='gpu' for DQNAgent, but CuPy / ROCm GPU runtime is not available."
            )

        self.rng = np.random.default_rng(seed)

        # Exploration parameters
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.eps_decay_steps = 6_000_000

        # Q update parameters
        self.gamma = 0.99
        self.steps = 0

        # Buffer and batching
        self.n_train_start = n_train_start
        self.n_target_start = n_target_start
        self.train_freq = train_freq
        self.batch_size = batch_size

        if obs_shape is None:
            if obs_dim is not None and obs_dim == 63:
                obs_shape = (1, 7, 9)
            elif isinstance(obs_dim, tuple):
                obs_shape = obs_dim
            elif obs_dim is not None:
                obs_shape = (obs_dim,)
            else:
                obs_shape = (3, 10, 10)

        self.obs_shape = obs_shape
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            obs_shape=obs_shape,
            obs_dtype=np.float32,
        )

        in_channels = obs_shape[0] if len(obs_shape) == 3 else 1

        def make_q_network(name):
            return Network(
                name,
                Sequential(
                    [
                        Conv2D(
                            in_channels=in_channels,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            input_shape=(in_channels, 22, 22),
                        ),
                        ReLU(),
                        Conv2D(
                            in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            input_shape=(16, 22, 22),
                        ),
                        ReLU(),
                        Conv2D(
                            in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            input_shape=(32, 11, 11),
                        ),
                        ReLU(),
                        Flatten(),
                        Linear(32 * 6 * 6, 320),
                        ReLU(),
                        Linear(320, 64),
                        ReLU(),
                        Linear(64, 4),
                    ]
                ),
                input_dim=in_channels * 22 * 22,
                output_dim=4,
            )

        self.Q_policy = make_q_network(f"{name}-policy")
        self.Q_target = make_q_network(f"{name}-target")

        self.Q_policy.to(device=self.device, dtype=self.dtype)
        self.Q_target.to(device=self.device, dtype=self.dtype)

        self.optimizer = ADAM(learning_rate=5e-4)
        self.optimizer.parameters = self.Q_policy.params()
        self.optimizer.to(device=self.device, dtype=self.dtype)

        self.network_update()

    @property
    def params(self) -> dict[str, np.ndarray]:
        """NumPy parameter dict for model artifact serialization."""
        return {f"p{i}": to_numpy(p.data) for i, p in enumerate(self.Q_policy.params())}

    @property
    def epsilon(self) -> float:
        return self.max_epsilon + min(1.0, self.steps / self.eps_decay_steps) * (
            self.min_epsilon - self.max_epsilon
        )

    def network_update(self) -> None:
        for t, p in zip(self.Q_target.params(), self.Q_policy.params()):
            t.data[...] = p.data

    def act_batch(
        self,
        observations: np.ndarray,
        greedy: bool = True,
    ) -> np.ndarray:
        """Execute batched policy-network inference across multiple observations.

        Args:
            observations: Batch of observations of shape (batch_size, *obs_shape).
            greedy: Whether to choose greedy actions or use epsilon-greedy exploration.

        Returns:
            np.ndarray: Integer array of actions on CPU of shape (batch_size,).
        """
        batch_size = len(observations)
        if batch_size == 0:
            return np.empty(0, dtype=np.int64)

        if greedy or self.rng.random() >= self.epsilon:
            obs_dev = to_device(observations, device=self.device, dtype=self.dtype)
            q_values = self.Q_policy.predict(obs_dev)
            xp = get_array_module(q_values)
            actions_dev = xp.argmax(q_values, axis=1)
            return to_numpy(actions_dev).astype(np.int64)
        else:
            return self.rng.integers(self.n_actions, size=batch_size, dtype=np.int64)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if greedy or self.rng.random() >= self.epsilon:
            obs_batch = obs[None, ...] if obs.ndim == len(self.obs_shape) else obs
            return int(self.act_batch(obs_batch, greedy=True)[0])
        else:
            # Explore: skip neural network inference
            return int(self.rng.integers(self.n_actions))

    def observe(self, obs, action, reward, next_obs, terminated) -> dict | None:
        self.replay_buffer.add(obs, action, reward, next_obs, terminated)
        self.steps += 1

        if (
            self.steps < self.n_train_start
            or len(self.replay_buffer) < self.batch_size * 100
            or self.steps % self.train_freq
        ):
            return None

        loss = dqn_update(
            self.Q_policy,
            self.Q_target,
            self.optimizer,
            self.replay_buffer,
            self.batch_size,
            self.gamma,
            self.rng,
            device=self.device,
            dtype=self.dtype,
        )

        if self.steps % self.n_target_start == 0:
            self.network_update()

        return {
            "loss": float(loss),
            "epsilon": self.epsilon,
        }


def save_checkpoint(step_dir: Path, agent: Any) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    agent.Q_policy.save_model(step_dir / "network")
    agent.optimizer.save_state(step_dir / "optimizer.npz", agent.Q_policy)


def _extract_step_from_path(path: Path | str) -> int:
    match = re.search(r"step_(\d+)", str(path))
    if match:
        return int(match.group(1))
    return 0


def restore_checkpoint(agent: DQNAgent, checkpoint_dir: Path | str) -> None:
    checkpoint_path = Path(checkpoint_dir)
    network_dir = checkpoint_path / "network"
    if not network_dir.exists() and (checkpoint_path / "manifest.json").exists():
        network_dir = checkpoint_path

    agent.Q_policy = Network.load_model(
        network_dir,
        device=agent.device,
        dtype=agent.dtype,
    )
    agent.optimizer.parameters = agent.Q_policy.params()
    optimizer_path = checkpoint_path / "optimizer.npz"
    agent.optimizer.load_state(optimizer_path, agent.Q_policy)
    agent.optimizer.to(device=agent.device, dtype=agent.dtype)
    agent.network_update()


def _is_wandb_disabled(run: Any) -> bool:
    if run is None:
        return True
    if getattr(run, "disabled", False) is True:
        return True
    settings = getattr(run, "settings", None)
    if settings is not None:
        mode = getattr(settings, "mode", None)
        if isinstance(mode, str) and mode == "disabled":
            return True
    mode = getattr(run, "mode", None)
    if isinstance(mode, str) and mode == "disabled":
        return True
    return False


def make_checkpoint_callback(
    checkpoint_interval: int = 100_000,
    wandb_checkpoint_interval: int = 1_000_000,
    checkpoints_dir: Path | str = "checkpoints",
):
    base_dir = Path(checkpoints_dir)
    run_id: str | None = None
    run_dir: Path | None = None
    saved_checkpoints: list[Path] = []
    last_saved_step: int | None = None
    last_uploaded_step: int | None = None

    def callback(
        step: int, agent: Any, run: Any = None, is_final: bool = False
    ) -> None:
        nonlocal run_id, run_dir, last_saved_step, last_uploaded_step

        if run_dir is None:
            run_id = getattr(run, "id", None) or f"run_{uuid.uuid4().hex[:8]}"
            run_dir = base_dir / run_id

        local_save_due = (
            checkpoint_interval > 0 and step % checkpoint_interval == 0
        ) or is_final
        wandb_enabled = not _is_wandb_disabled(run) and hasattr(run, "log_artifact")
        wandb_upload_due = (
            (wandb_checkpoint_interval > 0 and step % wandb_checkpoint_interval == 0)
            or is_final
        ) and wandb_enabled

        if not local_save_due and not wandb_upload_due:
            return

        step_dir = run_dir / f"step_{step}"

        # Save locally if not already saved for this step
        if step != last_saved_step:
            save_checkpoint(step_dir, agent)
            last_saved_step = step
            if step_dir not in saved_checkpoints:
                saved_checkpoints.append(step_dir)

        # Upload to W&B if due and not already uploaded for this step
        if wandb_upload_due and step != last_uploaded_step:
            artifact = wandb.Artifact(
                name=f"snake-dqn-{run.id}",
                type="model",
                metadata={
                    "run_id": run.id,
                    "step": step,
                    "_step": run.step,
                    "epsilon": agent.epsilon,
                },
            )
            artifact.add_dir(str(step_dir))
            logged_artifact = run.log_artifact(artifact)
            artifact_to_wait = (
                logged_artifact if logged_artifact is not None else artifact
            )
            if hasattr(artifact_to_wait, "wait") and callable(artifact_to_wait.wait):
                try:
                    artifact_to_wait.wait()
                except Exception:
                    pass
            last_uploaded_step = step

        # Retention: keep only the two most recent local checkpoints for this run
        while len(saved_checkpoints) > 2:
            oldest_dir = saved_checkpoints.pop(0)
            if oldest_dir.exists() and oldest_dir != step_dir:
                shutil.rmtree(oldest_dir, ignore_errors=True)

    return callback


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Snake Convolutional DQN Training")
    parser.add_argument(
        "--name",
        type=str,
        default="Snake-conv-DQN",
        help="Name of the network/run.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to train on (cpu or gpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for DQN updates.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=800_000,
        help="Total environment training steps.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5000,
        help="Evaluation interval in steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0xC0FFEE,
        help="Random seed.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100_000,
        help="Local checkpoint interval in steps (0 to disable).",
    )
    parser.add_argument(
        "--wandb-checkpoint-interval",
        type=int,
        default=1_000_000,
        help="W&B artifact checkpoint interval in steps (0 to disable).",
    )
    parser.add_argument(
        "--resume-local",
        type=str,
        default=None,
        help="Path to local checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--resume-wandb",
        type=str,
        default=None,
        help="W&B artifact reference to resume from.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["continue", "fork"],
        default=None,
        help="W&B resumption mode (continue or fork). Only applicable with --resume-wandb.",
    )
    parsed = parser.parse_args(args)
    if parsed.resume_local and parsed.resume_wandb:
        parser.error("--resume-local and --resume-wandb are mutually exclusive.")
    if parsed.wandb_mode is not None and not parsed.resume_wandb:
        parser.error(
            "--wandb-mode is only applicable when resuming from W&B (--resume-wandb)."
        )
    return parsed


def main(args=None):
    if args is None:
        args = parse_args()
    elif isinstance(args, list):
        args = parse_args(args)

    selected_dtype = np.float32 if args.dtype == "float32" else np.float64
    env = make_snake_env()
    dqn_agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=args.seed,
        device=args.device,
        dtype=selected_dtype,
        batch_size=args.batch_size,
        name=args.name,
    )

    start_step = 0
    run_id: str | None = None
    resume: str | None = None
    fork_from: str | None = None
    use_artifact: Any | None = None

    if args.resume_local:
        checkpoint_dir = Path(args.resume_local)
        restore_checkpoint(dqn_agent, checkpoint_dir)
        start_step = _extract_step_from_path(checkpoint_dir)
        dqn_agent.steps = start_step
    elif args.resume_wandb:
        api = wandb.Api()
        artifact = api.artifact(args.resume_wandb)
        download_dir = artifact.download()
        checkpoint_dir = Path(download_dir)
        restore_checkpoint(dqn_agent, checkpoint_dir)

        metadata = artifact.metadata or {}
        start_step = int(metadata.get("step", _extract_step_from_path(checkpoint_dir)))
        dqn_agent.steps = start_step

        orig_run_id = artifact.logged_by().id
        use_artifact = artifact

        wandb_mode = args.wandb_mode or "continue"
        if wandb_mode == "continue":
            run_id = orig_run_id
            resume = "must"
        elif wandb_mode == "fork":
            fork_step = artifact.metadata["_step"]
            fork_from = f"{orig_run_id}?_step={fork_step}"

    checkpoint_callback = make_checkpoint_callback(
        checkpoint_interval=args.checkpoint_interval,
        wandb_checkpoint_interval=args.wandb_checkpoint_interval,
    )

    train(
        dqn_agent,
        make_env=make_snake_env,
        n_steps=args.n_steps,
        eval_every=args.eval_every,
        group="Convolution",
        name=args.name,
        config={
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
        },
        make_video_env=lambda: RenderedSnakeEnv(
            width=WIDTH,
            height=HEIGHT,
            max_steps_no_food=WIDTH * HEIGHT * 2,
            rewards=REWARDS,
            render_mode="rgb_array",
            obs_mode="board",
            movement_rule=True,
        ),
        checkpoint_callback=checkpoint_callback,
        start_step=start_step,
        run_id=run_id,
        resume=resume,
        fork_from=fork_from,
        use_artifact=use_artifact,
    )


if __name__ == "__main__":
    main()
