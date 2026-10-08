import gymnasium as gym
from .cart_pole import CartPoleAgent

from src.training.dqn_harness import evaluate, train


class TrainAgent:
    def __init__(
        self,
        env,
        agent,
        learning_rate,
        n_episodes,
        start_epsilon,
        epsilon_decay,
        final_epsilon,
    ):
        self.lr = learning_rate
        self.n_episodes = n_episodes
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.env = env
        self.agent = agent

        self.train_agent()

    def train_agent(self):

        for episode in range(self.n_episodes):
            obs, info = self.env.reset()
            done = False

            while not done:
                action = self.agent.act(obs, True)

                next_obs, reward, terminated, truncated, info = self.env.step(action)

                self.agent.observe(obs, action, reward, terminated, next_obs)

                done = terminated

                obs = next_obs

        self.agent.decay_epsilon()


def make_env():
    return gym.make("CartPole-v1", render_mode="human")


def main():
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = gym.make("CartPole-v1", render_mode="human")

    agent = CartPoleAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    ta = TrainAgent(
        env,
        agent,
        learning_rate,
        n_episodes,
        start_epsilon,
        epsilon_decay,
        final_epsilon,
    )

    # ta.train_agent()

    train(
        agent=agent,
        make_env=make_env(),
        n_steps=5000,
    )

    evaluate(agent=agent, make_env=make_env(), n_episodes=100, seed=10_000)


if __name__ == "__main__":
    main()
