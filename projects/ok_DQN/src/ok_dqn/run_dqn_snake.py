from coppercore.dqn.snake_env import SnakeEnv
from coppercore.training.dqn_harness import train
from ok_dqn.agent.DQN_agent import DQNAgent


def make_snake_env():
    return SnakeEnv(width=5, height=5, seed=0, max_steps_no_food=100)


env = SnakeEnv()
n_actions = env.n_actions
n_states = env.n_states

q_agent = DQNAgent(
    action_space=n_actions, state_dim=1, buffer_size=10_000, sample_size=32
)

train(
    q_agent,
    make_snake_env,
    1_500_000,
    eval_every=2_000,
    group="OK",
    name="OK-ql",
    config={"alpha": q_agent.optimizer.learning_rate, "gamma": q_agent.gamma},
)
