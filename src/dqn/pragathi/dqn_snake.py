import numpy as np
import collections 
import random
from src.dqn.snake_env import SnakeEnv, Action
from src.training.dqn_harness import evaluate, train


class Parameter:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = np.zeros_like(data)

def default_init(rng: np.random.Generator, shape: tuple, lim) -> np.ndarray:
    return rng.uniform(-lim, lim, shape)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def update_buffer(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))
    
    def create_sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # turns list of tuples into separate tuples 
        states, actions, next_states, rewards, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float64),
            np.asarray(actions, dtype=np.int64),
            np.asarray(next_states, dtype=np.float64),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(dones, dtype=np.float64),
        )

class QNetwork:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        rng,
        hidden_dim=128
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        w1_shape = (state_dim, hidden_dim)
        b1_shape = (hidden_dim,)
        w2_shape = (hidden_dim, action_dim)
        b2_shape = (action_dim,)

        # Initiating weights and biases
        self.w1 =  Parameter(default_init(rng, w1_shape, 1))
        self.b1 =  Parameter(default_init(rng, b1_shape, 1))

        self.w2 = Parameter(default_init(rng, w2_shape, 1))
        self.b2 = Parameter(default_init(rng, b2_shape, 1))

    def forward(self, x):
        self.x = x

        self.z1 = self.x @ self.w1.data + self.b1.data
        self.h1 = self.reLu(self.z1)

        self.q_values = self.h1 @ self.w2.data + self.b2.data

        return self.q_values

    def backward(self, grad_output):
        batch_size = self.x.shape[0]

        self.w2.grad = self.h1.T @ grad_output
        self.b2.grad = np.sum(grad_output, axis=0)

        grad_hidden = grad_output @ self.w2.data.T

        grad_hidden *= self.mask

        self.w1.grad = self.x.T @ grad_hidden
        self.b1.grad = np.sum(grad_hidden, axis=0)

    def reLu(self, x: np.ndarray):
        out = np.maximum(x, 0)
        self.mask = out > 0
        return out
        
class SnakeDQN:
    def __init__(
        self, 
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        seed: int,
        state_dim:int, 
        action_dim:int,
        replay_buffer:ReplayBuffer,
        discount_factor: float = 0.95, 

    ):
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(seed)
        self.replay_buffer = replay_buffer

        self.batch_size = 64
        self.warmup = 1000          
        self.target_update = 500    
        self.steps = 0

        # Random rumber for q_network and target network
        rng = np.random.default_rng(seed=42)

        # Q network and target network initialisation
        self.q_network = QNetwork(state_dim, action_dim, rng)
        self.target_network = QNetwork(state_dim, action_dim, rng)
        self.update_target_network()

        # Array of training errors
        self.training_error = []


    def act(self, obs: np.ndarray, greedy: bool = False):
        self.num_actions = action_dim

        if greedy:
            return np.argmax(self.q_network[obs]) # Exploit

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))  # Explore

        return int(np.argmax(self.q_network.forward(obs)))


    def observe(self, obs, action, reward, next_obs, done):
        # update the replay buffer 
        self.replay_buffer.update_buffer(state, action, next_state, reward, done)
        
    def update_target(self):
        self.target_network.w1.data = self.q_network.w1.data.copy()
        self.target_network.b1.data = self.q_network.b1.data.copy()

        self.target_network.w2.data = self.q_network.w2.data.copy()
        self.target_network.b2.data = self.q_network.b2.data.copy()

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def make_env():
    env = SnakeEnv(width=10, height=10, seed=0)
    obs, info = env.reset()
    return env

class main():
    rb = ReplayBuffer(10**6)
    learning_rate = 0.1
    n_episodes = 50_000
    start_epsilon = 0.9
    final_epsilon = 0.02
    epsilon_decay = (start_epsilon - final_epsilon) / (n_steps / 2)
    action_dim = 5
    state_dim = 4

    env = make_env()

    snake_dqn = SnakeDQN(
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        seed=42,
        state_dim=state_dim, 
        action_dim=action_dim,
        replay_buffer=rb,
    )

    train(
        agent=snake_dqn,
        make_env=make_env,
        n_steps=n_steps,
        make_video_env=lambda: RenderedSnakeEnv(
            width=10, height=10, max_steps_no_food=200, render_mode="rgb_array"
        ),
    )

    evaluate(agent=snake_dqn, make_env=make_env)

if __name__ == "__main__":
    main()