import gymnasium as gym
from coppercore.training.dqn_harness import train
from ok_dqn.agent.DQN_agent import DQNAgent


def make_cartpole():
    return gym.make("CartPole-v1")


env = make_cartpole()
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

dqn_agent = DQNAgent(
    action_space=n_actions, state_dim=state_dim, buffer_size=10_000, sample_size=32
)

train(
    dqn_agent,
    make_cartpole,
    50_000,
    eval_every=1_000,
    group="OK",
    name="OK-dqn_cartpole",
    config={"alpha": dqn_agent.optimizer.learning_rate, "gamma": dqn_agent.gamma},
)
