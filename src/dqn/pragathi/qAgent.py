from coppercore.dqn.snake_env import SnakeEnv, Action
from coppercore.training.dqn_harness import evaluate, train
# from coppercore.dqn.snake_render import RenderedSnakeEnv
import numpy as np

class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        seed: int,
        n_states:int,
        n_actions:int,
        discount_factor: float = 0.95, 
    ):
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(seed)

        #Initialise Q - table 
        self.num_states = n_states
        self.num_actions = n_actions
    
        self.qtable = np.zeros((self.num_states, self.num_actions))
        self.training_error = []
        

    def act(self, obs: np.ndarray, greedy: bool = False):
        if greedy:
            return np.argmax(self.qtable[obs]) # Exploit

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))  # Explore

        return np.argmax(self.qtable[obs]) 

    # TODO: DQN used here to predict the Q - values 
    def observe(self, obs, action, reward, next_obs, terminated):
        future_q_value = (not terminated) * np.max(self.qtable[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.qtable[obs][action]

        self.qtable[obs][action] = (
            self.qtable[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)
        self.decay_epsilon()

        # training requires a dict 
        return {
            "q_value": self.qtable[obs, action],
        }

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def make_env():
    env = SnakeEnv(width=10, height=10, seed=0)
    obs, info = env.reset()
    return env

def main():
    learning_rate = 0.01
    start_epsilon = 0.9
    n_steps = 50_000
    final_epsilon = 0.02
    epsilon_decay = (start_epsilon - final_epsilon) / (n_steps / 2)


    env = make_env()
    agent = QLearningAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        seed=42,
        n_states=env.n_states,
        n_actions=env.n_actions,
    )

    train(
        agent=agent,
        make_env=make_env,
        n_steps=n_steps,
        make_video_env=Lamd
    )

    evaluate(agent=agent, make_env=make_env)


if __name__ == "__main__":
    main()