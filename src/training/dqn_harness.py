from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import wandb


class Agent(Protocol):
    def act(self, obs: np.ndarray, greedy: bool = False) -> int: ...
    def observe(self, obs, action, reward, next_obs, terminated) -> dict | None: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...


def evaluate(
    agent: Agent, make_env: Callable, n_episodes: int = 20, seed: int = 10_000
) -> dict:
    """Play n_episodes without exploration (greedy=True) and return the averages."""
    # a separate env, so evaluation doesn't interrupt the training episode
    env = make_env()
    returns, scores = [], []

    for i in range(n_episodes):
        # same seeds every time, so results are comparable across agents and runs
        obs, info = env.reset(seed=seed + i)
        episode_return = 0.0
        done = False

        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)

        if "score" in info:
            scores.append(info["score"])

    out = {"return_mean": float(np.mean(returns))}
    if scores:
        out["score_mean"] = float(np.mean(scores))
    return out


def train(
    agent: Agent,
    make_env: Callable,
    n_steps: int,
    seed: int = 0,
    config=None,
    group=None,
    name=None,
    log_every=1_000,
    eval_every=10_000,
):
    """Train the agent for n_steps environment steps and log everything to WandB."""

    run = wandb.init(
        project="copperhead-rl",
        entity=None,
        group=group,  # seeds of the same setup, like "dqn-cartpole"
        name=name,
        config={"seed": seed, "n_steps": n_steps, **(config or {})},
    )

    # the training env, only the first reset is seeded
    env = make_env()
    obs, info = env.reset(seed=seed)
    episode_return, episode_length = 0.0, 0
    best_eval = float("-inf")

    for step in range(1, n_steps + 1):
        # choose an action (with exploration)
        action = agent.act(obs)

        # environment response
        next_obs, reward, terminated, truncated, info = env.step(action)

        # agent learns, only terminated is passed, since a truncation is not a terminal state
        metrics = agent.observe(obs, action, reward, next_obs, terminated)

        obs = next_obs
        episode_return += reward
        episode_length += 1

        # log episode when it's over (died or timed out)
        if terminated or truncated:
            log = {"episode/return": episode_return, "episode/length": episode_length}

            if "score" in info:
                log["episode/score"] = info["score"]

            run.log(log, step=step)
            obs, info = env.reset()
            episode_return, episode_length = 0.0, 0

        # log the agent's data (latest value, use smoothing in WandB for noisy curves)
        if metrics and step % log_every == 0:
            run.log({f"agent/{k}": v for k, v in metrics.items()}, step=step)

        # evaluate agent occasionally, and keep the best version
        if step % eval_every == 0 or step == n_steps:
            result = evaluate(agent, make_env)
            run.log({f"eval/{k}": v for k, v in result.items()}, step=step)

            if result["return_mean"] > best_eval:
                best_eval = result["return_mean"]
                agent.save(f"{run.dir}/best.npz")
                run.summary["best_eval_return"] = best_eval

    # upload the best weights as a WandB artifact
    artifact = wandb.Artifact(f"agent-{run.id}", type="model")
    artifact.add_file(f"{run.dir}/best.npz")
    run.log_artifact(artifact)
    run.finish()
