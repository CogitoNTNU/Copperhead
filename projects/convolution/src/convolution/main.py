import numpy as np

from coppercore.dqn.base_agent import BaseAgent
from coppercore.dqn.snake_env import SnakeEnv, Event
from coppercore.training.dqn_harness import train
from coppercore.dqn.snake_render import RenderedSnakeEnv

from coppercore.training.dqn_train import dqn_update

from coppercore.buffer import ReplayBuffer
from coppercore.model_driven_nn.network import Network
from coppercore.model_driven_nn.components import (
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    Sequential,
)
from coppercore.optimizer import ADAM


REWARDS = {Event.MOVED: -0.001, Event.ATE: 1.0, Event.DIED: -1.0, Event.WON: 10.0}

WIDTH = 8
HEIGHT = 8


def make_snake_env():
    return SnakeEnv(
        width=WIDTH,
        height=HEIGHT,
        seed=0,
        max_steps_no_food=WIDTH * HEIGHT * 2,
        rewards=REWARDS,
        obs_mode="board",
        obs_window_radius=2,
    )


class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        obs_dim: int | None = None,
        obs_shape: tuple[int, ...] | None = None,
        seed: int = 0,
    ):
        self.n_state = n_states
        self.n_actions = n_actions

        self.rng = np.random.default_rng(seed)

        # Exploration parameters
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.eps_decay_steps = 400_000

        # Q update parameters
        self.gamma = 0.99
        self.steps = 0

        # Buffer and batching
        self.n_train_start = 2000
        self.n_target_start = 1000
        self.train_freq = 4

        self.batch_size = 100

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
            capacity=self.eps_decay_steps, obs_shape=obs_shape, obs_dtype=np.float32
        )

        in_channels = obs_shape[0] if len(obs_shape) == 3 else 1
        conv_h, conv_w = (
            (obs_shape[1], obs_shape[2])
            if len(obs_shape) == 3
            else (
                (obs_shape[0], obs_shape[1])
                if len(obs_shape) == 2
                else (1, obs_shape[0])
            )
        )
        conv_out_channels = 16
        conv_flat_dim = conv_out_channels * conv_h * conv_w
        total_dim = int(np.prod(obs_shape))

        self.Q_policy = Network(
            "snake-dqn-policy",
            Sequential(
                [
                    Conv2D(
                        in_channels=in_channels,
                        out_channels=conv_out_channels,
                        kernel_size=3,
                        padding=1,
                        input_shape=(in_channels, conv_h, conv_w),
                    ),
                    ReLU(),
                    Flatten(),
                    Linear(conv_flat_dim, 64),
                    ReLU(),
                    Linear(64, 64),
                    ReLU(),
                    Linear(64, 4),
                ]
            ),
            input_dim=total_dim,
            output_dim=4,
        )
        self.Q_target = Network(
            "snake-dqn-target",
            Sequential(
                [
                    Conv2D(
                        in_channels=in_channels,
                        out_channels=conv_out_channels,
                        kernel_size=3,
                        padding=1,
                        input_shape=(in_channels, conv_h, conv_w),
                    ),
                    ReLU(),
                    Flatten(),
                    Linear(conv_flat_dim, 64),
                    ReLU(),
                    Linear(64, 64),
                    ReLU(),
                    Linear(64, 4),
                ]
            ),
            input_dim=total_dim,
            output_dim=4,
        )

        self.optimizer = ADAM(learning_rate=5e-4)
        self.optimizer.parameters = self.Q_policy.params()

        # The weights and bias for the policy network
        self.params = {f"p{i}": p.data for i, p in enumerate(self.Q_policy.params())}
        self.network_update()

    @property
    def epsilon(self) -> float:
        return self.max_epsilon + min(1.0, self.steps / self.eps_decay_steps) * (
            self.min_epsilon - self.max_epsilon
        )

    def network_update(self) -> None:
        for t, p in zip(self.Q_target.params(), self.Q_policy.params()):
            t.data[...] = p.data

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if greedy or self.rng.random() >= self.epsilon:
            # exploit and choose the current best action
            obs_batch = obs[None, ...] if obs.ndim == len(self.obs_shape) else obs
            q_values = self.Q_policy.forward(obs_batch)
            return int(np.argmax(q_values))
        else:
            # explore other actions
            # TODO: remove greedy action from selection
            return int(self.rng.integers(self.n_actions))

    def observe(self, obs, action, reward, next_obs, terminated) -> dict | None:
        self.replay_buffer.add(obs, action, reward, next_obs, terminated)
        self.steps += 1

        if len(self.replay_buffer) < self.n_train_start or self.steps % self.train_freq:
            return None

        loss = dqn_update(
            self.Q_policy,
            self.Q_target,
            self.optimizer,
            self.replay_buffer,
            self.batch_size,
            self.gamma,
            self.rng,
        )

        if self.steps % self.n_target_start == 0:
            self.network_update()

        return {
            "loss": loss,
            "epsilon": self.epsilon,
        }  # a returned dict logged in WandB


if __name__ == "__main__":
    env = make_snake_env()
    dqn_agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        obs_shape=env.obs_shape,
        seed=0xC0FFEE,
    )

    train(
        dqn_agent,
        make_env=make_snake_env,
        n_steps=800_000,
        eval_every=5000,
        group="Convolution",
        name="Snake-conv-DQN",
        make_video_env=lambda: RenderedSnakeEnv(
            width=WIDTH,
            height=HEIGHT,
            max_steps_no_food=WIDTH * HEIGHT * 2,
            rewards=REWARDS,
            render_mode="rgb_array",
            obs_mode="board",
        ),
    )
