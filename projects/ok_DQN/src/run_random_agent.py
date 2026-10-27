from projects.ok_DQN.src.agent.random_agent import RandomAgent
from packages.coppercore.src.coppercore.training.dqn_harness import train
from packages.coppercore.src.coppercore.dqn.snake_env import SnakeEnv


def make_snake_env():
    return SnakeEnv(width=5, height=5, seed=0, max_steps_no_food=100)


env = SnakeEnv()
n_actions = env.n_actions
n_states = env.n_states

random = RandomAgent(n_actions)
train(random, make_snake_env, 10000, group="OK", name="OK")
