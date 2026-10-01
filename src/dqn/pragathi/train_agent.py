import gymnasium as gym
import numpy as np

class TrainAgent:
    def __init__(
       #TODO
    ):


env = gym.make('CartPole-v1', render_mode="human")

# Training Hyperparameters 
learning_rate = 0.01
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = CartPoleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
