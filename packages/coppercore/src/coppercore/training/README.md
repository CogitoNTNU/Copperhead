# dqn_harness

A shared training loop for reinforcement learning agents. You provide an agent and an environment, and `train()` runs the training, evaluates the agent, logs to WandB and saves the best weights. This makes results comparable across agents.

## Usage

```python
from coppercore.dqn.base_agent import BaseAgent
from coppercore.dqn.snake_env import SnakeEnv
from coppercore.training.dqn_harness import train


class MyAgent(BaseAgent):
    def act(self, obs, greedy=False):
        # greedy=True during evaluation, so turn off exploration
        return int(self.rng.integers(self.n_actions))

    def observe(self, obs, action, reward, next_obs, terminated):
        # learn from the step here
        return {"loss": 0.0}  # logged as agent/loss


def make_env():
    return SnakeEnv(width=5, height=5)


agent = MyAgent(make_env().n_actions)
train(agent, make_env, n_steps=10_000, group="my-agent-snake", name="seed-0")
```

Runs are logged to the `copperhead-rl` project in the `Copperhead` WandB team, so you need access to it. To try the harness without logging in, set the environment variable `WANDB_MODE=offline`.

## What your agent needs

An agent does not have to inherit from `BaseAgent`, as long as it has these two methods:

| Method                                               | Purpose                                                                     |
| ---------------------------------------------------- | --------------------------------------------------------------------------- |
| `act(obs, greedy=False) -> int`                      | Choose an action. `greedy=True` means no exploration.                       |
| `observe(obs, action, reward, next_obs, terminated)` | Learn from one step. Can return a dict of metrics to log, otherwise `None`. |

To save your weights, also give the agent a `params` attribute: a dict of NumPy arrays. Without it, the saved file is empty.

`observe` only gets `terminated`, not `truncated`, since an episode that is cut off is not a real terminal state. Keep bootstrapping from `next_obs` in that case.

## What `train()` does

1. Starts a WandB run.
1. Plays `n_steps` environment steps, calling `act` and `observe` on every step and resetting the environment when an episode ends.
1. Every `eval_every` steps and at the last step, plays 20 greedy episodes on a fresh environment. The seeds are always the same, so the results can be compared across runs.
1. Saves `params` to `best.npz` whenever the mean evaluation return improves, and uploads the best version as a WandB artifact at the end.

## Arguments

| Argument         | Default  | Description                                                                                                                     |
| ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `agent`          | required | The agent to train.                                                                                                             |
| `make_env`       | required | Function that returns a new environment.                                                                                        |
| `n_steps`        | required | Number of environment steps to train for.                                                                                       |
| `seed`           | `0`      | Seed for the first reset of the training environment. Does not seed the agent.                                                  |
| `config`         | `None`   | Dict of hyperparameters to store with the run.                                                                                  |
| `group`          | `None`   | WandB group. Use the same group for several seeds of one setup.                                                                 |
| `name`           | `None`   | WandB run name.                                                                                                                 |
| `log_every`      | `1_000`  | How often to log the agent's metrics, in steps.                                                                                 |
| `eval_every`     | `10_000` | How often to evaluate, in steps.                                                                                                |
| `make_video_env` | `None`   | Function that returns a `RenderedSnakeEnv` with `render_mode="rgb_array"`. Logs a gif of the best agent. Only works with Snake. |

## What gets logged

| Key                 | Description                                             |
| ------------------- | ------------------------------------------------------- |
| `episode/return`    | Total reward in a training episode.                     |
| `episode/length`    | Number of steps in the episode, not the snake's length. |
| `episode/score`     | `info["score"]`, if the environment provides it.        |
| `agent/*`           | The metrics returned by `observe`.                      |
| `eval/return_mean`  | Mean return over the evaluation episodes.               |
| `eval/score_mean`   | Mean `info["score"]`, if the environment provides it.   |
| `eval/best_episode` | Gif of the best agent, if `make_video_env` is set.      |
