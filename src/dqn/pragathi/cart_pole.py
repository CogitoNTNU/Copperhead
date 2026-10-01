import numpy as np

class CartPoleAgent:

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay:float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdoct(lambda: np.zeroes(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters: how much exploration are we going to do
        self.epsilon = initial_epsilon
        self.epislon.decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []


    def get_random_action(self, observation: tuple[float, float, flaot, fl]):
        




