import numpy as np

from coppercore.dqn.base_agent import BaseAgent
from coppercore.dqn.snake_env import SnakeEnv, Event
from coppercore.training.dqn_harness import train
from coppercore.dqn.snake_render import RenderedSnakeEnv

from coppercore.training.dqn_train import dqn_update

from coppercore.buffer import ReplayBuffer
from coppercore.model_driven_nn.network import Network
from coppercore.model_driven_nn.components import Sequential, Linear, ReLU
from coppercore.optimizer import ADAM


REWARDS = {Event.MOVED: -0.001, Event.ATE: 1.0, Event.DIED: -1.0, Event.WON: 10.0}


def make_snake_env():
    return SnakeEnv(
        width=10,
        height=10,
        seed=0,
        max_steps_no_food=100,
        rewards=REWARDS,
        obs_mode="obs",
        obs_window_radius=2,
    )


class DQNAgent(BaseAgent):
    def __init__(self, n_states: int, n_actions: int, obs_dim: int, seed: int = 0):
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
        self.replay_buffer = ReplayBuffer(
            capacity=self.eps_decay_steps, obs_shape=(obs_dim,), obs_dtype=np.float32
        )

        self.Q_policy = Network(
            "snake-dqn-policy",
            Sequential(
                [Linear(obs_dim, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 4)]
            ),
            input_dim=obs_dim,
            output_dim=4,
        )
        self.Q_target = Network(
            "snake-dqn-target",
            Sequential(
                [Linear(obs_dim, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 4)]
            ),
            input_dim=obs_dim,
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

    def act_batch(
        self,
        observations: np.ndarray,
        greedy: bool = True,
    ) -> np.ndarray:
        batch_size = len(observations)
        if batch_size == 0:
            return np.empty(0, dtype=np.int64)
        if greedy or self.rng.random() >= self.epsilon:
            q_values = self.Q_policy.predict(observations)
            return np.argmax(q_values, axis=1).astype(np.int64)
        else:
            return self.rng.integers(self.n_actions, size=batch_size, dtype=np.int64)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if greedy or self.rng.random() >= self.epsilon:
            obs_batch = obs[None, ...] if obs.ndim == 1 else obs
            return int(self.act_batch(obs_batch, greedy=True)[0])
        else:
            # explore other actions
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
        n_states=env.n_states, n_actions=env.n_actions, obs_dim=env.obs_dim, seed=42
    )

    train(
        dqn_agent,
        make_env=make_snake_env,
        n_steps=800_000,
        eval_every=5000,
        group="fred_test",
        name="Snake-DQN",
        make_video_env=lambda: RenderedSnakeEnv(
            width=10,
            height=10,
            max_steps_no_food=100,
            rewards=REWARDS,
            render_mode="rgb_array",
            obs_mode="obs",
        ),
    )
