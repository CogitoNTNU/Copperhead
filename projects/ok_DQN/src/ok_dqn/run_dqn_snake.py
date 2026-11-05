from coppercore.dqn.snake_env import SnakeEnv
from coppercore.training.dqn_harness import train
from ok_dqn.agent.DQN_agent import DQNAgent
from coppercore.dqn.snake_render import RenderedSnakeEnv


def make_snake_env():
    return SnakeEnv(width=5, height=5, seed=0, max_steps_no_food=100, obs_mode="obs")


def make_video_env():
    return RenderedSnakeEnv(
        width=5,
        height=5,
        seed=0,
        max_steps_no_food=100,
        obs_mode="obs",
        render_mode="rgb_array",
    )


env = make_snake_env()
n_actions = env.n_actions
n_states = env.n_states
state_dim = env.reset()[0].size
dqn_agent = DQNAgent(
    action_space=n_actions,
    state_dim=state_dim,
    buffer_size=20_000,
    sample_size=64,
    decay_eps=0.9995,
)

train(
    dqn_agent,
    make_snake_env,
    n_steps=250_000,
    eval_every=2_000,
    group="OK",
    name="OK-dqn-snake",
    config={"alpha": dqn_agent.optimizer.learning_rate, "gamma": dqn_agent.gamma},
    make_video_env=make_video_env,
)
