from packages.coppercore.src.coppercore.training.dqn_harness import train
from packages.coppercore.src.coppercore.dqn.snake_env import SnakeEnv
from projects.ok_DQN.src.agent.Q_agent import QAgent


def make_snake_env():
    return SnakeEnv(width=5, height=5, seed=0, max_steps_no_food=100)


env = SnakeEnv()
n_actions = env.n_actions
n_states = env.n_states

ql_agent = QAgent(n_states, n_actions)

train(
    ql_agent,
    make_snake_env,
    1_500_000,
    eval_every=2_000,
    group="OK",
    name="OK-ql",
    config={
        "alpha": ql_agent.alpha,
        "gamma": ql_agent.gamma,
        "epsilon_decay": ql_agent.epsilon_decay,
    },
)
