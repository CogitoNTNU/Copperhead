from src.dqn.snake_env import SnakeEnv, Action
from src.training.dqn_harness import evaluate, train

import numpy as np

class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        seed: int,
        discount_factor: float = 0.95, 
    ):
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(seed)

        #Initialise Q - table 
        self.num_states = 4
        self.num_actions = 4
    
        self.qtable = np.zeros((self.num_states, self.num_actions))
        self.training_error = []
        

    def act(self, obs: np.ndarray, greedy: bool = False):
        if greedy:
            return np.argmax(self.qtable[obs]) # Exploit

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))  # Explore

        return np.argmax(self.qtable[obs]) 

    def observe(self, obs, action, reward, next_obs, terminated):
        print(next_obs)
        # TODO: Find suitable implementation/interpretation of the observation to update the q_value.
            # divide the vector into three instead 
            # convert an observation to a valid event
        future_q_value = (not terminated) * np.max(self.qtable[next_obs])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.qtable[obs][action]

        self.qtable[obs][action] = (
            self.qtable[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)
        self.decay_epsilon()
        return self.qtable

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def make_env():
    env = SnakeEnv(width=10, height=10, seed=0)
    obs, info = env.reset()
    env.render()
    return env

def main():
    learning_rate = 0.1
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = QLearningAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        seed=42,
    )

    train(
        agent=agent,
        make_env=make_env,
        n_steps=42,
    )

    evaluate(agent=agent, make_env=make_env)


if __name__ == "__main__":
    main()