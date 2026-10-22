from src.dqn.snake_env import SnakeEnv
from projects.OK_dqn.agent.q_agent import QAgent
from projects.OK_dqn.OK_harness import train


def make_snake_env():
    return SnakeEnv(width=5, height=5, seed=0, max_steps_no_food=100)


env = SnakeEnv()
n_actions = env.n_actions
n_states = env.n_states

ql_agent = QAgent(n_states, n_actions)

train(
    ql_agent,
    make_snake_env,
    150_000,
    eval_every=200,
    name="OK",
)
