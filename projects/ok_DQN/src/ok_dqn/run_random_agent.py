from coppercore.dqn.snake_env import SnakeEnv
from coppercore.training.dqn_harness import train
from ok_dqn.agent.random_agent import RandomAgent


def make_snake_env():
    return SnakeEnv(width=5, height=5, seed=0, max_steps_no_food=100)


def main():
    my_agent = RandomAgent(make_snake_env().action_space)
    train(my_agent, make_snake_env, 10_000, group="OK", name="OK")


if __name__ == "__main__":
    main()
