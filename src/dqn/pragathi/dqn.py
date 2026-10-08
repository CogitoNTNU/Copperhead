import numpy as np
import gymnasium as gym
import random 
from collections import defaultdict 
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Optional

def make_env(
    env_name: str
):
    env = gym.make(env_name)
    return env

def argmax(a):
    a = np.array(a)
    return n.random.choice(np.arrange(len(a), dtype=int)[a == np.max(a)])

def Qlearn(
    make_env,
    learning_rate=0.2,
    discount_factor=0.95,
    final_epsilon=0.1,
    n_steps=800_000,
    callback_freq=5000,
    callback=None
):
    episode_reward = 0
    mean_episodic_reward = 0
    n_episodes = 0

    epsilon = 1.0