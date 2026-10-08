import gymnasium as gym
from .cart_pole import CartPoleAgent

from src.training.dqn_harness import evaluate, train

def make_env():
    return gym.make("CartPole-v1")

def main():
    learning_rate = 0.1
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = CartPoleAgent(
        env=make_env(),
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    train(
        agent=agent,
        make_env=make_env,
        n_steps=42,
    )

    evaluate(agent=agent, make_env=make_env)

if __name__ == "__main__":
    main()
