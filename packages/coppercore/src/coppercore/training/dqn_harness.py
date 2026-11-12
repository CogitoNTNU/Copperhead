import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import wandb
from tqdm import trange


# required agent structure
class Agent(Protocol):
    def act(self, obs: np.ndarray, greedy: bool = False) -> int: ...

    def observe(self, obs, action, reward, next_obs, terminated) -> dict | None: ...


def evaluate(
    agent: Agent, make_env: Callable, n_episodes: int = 20, seed: int = 10_000
) -> dict:
    """Evaluate agent across n_episodes using batched inference and return metrics."""
    if n_episodes <= 0:
        return {
            "return_mean": 0.0,
            "duration_seconds": 0.0,
            "total_episodes": 0,
            "total_transitions": 0,
            "env_steps_per_second": 0.0,
            "inference_calls": 0,
            "mean_inference_batch_size": 0.0,
            "mean_inference_latency_seconds": 0.0,
            "inference_duration_seconds": 0.0,
            "episode_length_mean": 0.0,
        }

    start_time = time.perf_counter()
    envs = [make_env() for _ in range(n_episodes)]

    current_obs = [None] * n_episodes
    episode_returns = [0.0] * n_episodes
    episode_lengths = [0] * n_episodes
    episode_scores = [None] * n_episodes

    for i in range(n_episodes):
        obs, info = envs[i].reset(seed=seed + i)
        current_obs[i] = obs
        if "score" in info:
            episode_scores[i] = info["score"]

    active_indices = list(range(n_episodes))
    total_transitions = 0
    inference_calls = 0
    total_inference_batch_size = 0
    total_inference_time = 0.0

    has_act_batch = callable(getattr(agent, "act_batch", None))

    while active_indices:
        batch_obs = np.stack([current_obs[i] for i in active_indices], axis=0)
        batch_size = len(active_indices)
        total_inference_batch_size += batch_size
        inference_calls += 1

        t0 = time.perf_counter()
        if has_act_batch:
            actions = agent.act_batch(batch_obs, greedy=True)
        else:
            actions = np.array(
                [agent.act(current_obs[i], greedy=True) for i in active_indices],
                dtype=np.int64,
            )
        total_inference_time += time.perf_counter() - t0

        next_active_indices = []
        for idx, env_idx in enumerate(active_indices):
            action = int(actions[idx])
            obs, reward, terminated, truncated, info = envs[env_idx].step(action)
            episode_returns[env_idx] += float(reward)
            episode_lengths[env_idx] += 1
            total_transitions += 1

            if "score" in info:
                episode_scores[env_idx] = info["score"]

            if not (terminated or truncated):
                current_obs[env_idx] = obs
                next_active_indices.append(env_idx)

        active_indices = next_active_indices

    for env in envs:
        if hasattr(env, "close") and callable(env.close):
            env.close()

    total_duration = time.perf_counter() - start_time

    out = {
        "return_mean": float(np.mean(episode_returns)),
        "duration_seconds": float(total_duration),
        "total_episodes": n_episodes,
        "total_transitions": total_transitions,
        "env_steps_per_second": (
            float(total_transitions / total_duration) if total_duration > 0 else 0.0
        ),
        "inference_calls": inference_calls,
        "mean_inference_batch_size": (
            float(total_inference_batch_size / inference_calls)
            if inference_calls > 0
            else 0.0
        ),
        "mean_inference_latency_seconds": (
            float(total_inference_time / inference_calls)
            if inference_calls > 0
            else 0.0
        ),
        "inference_duration_seconds": float(total_inference_time),
        "episode_length_mean": float(np.mean(episode_lengths)),
    }

    scores = [s for s in episode_scores if s is not None]
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
    make_video_env: Callable | None = None,
):
    """Train the agent for n_steps environment steps and log everything to WandB."""

    run = wandb.init(
        project="copperhead-rl",
        entity="Copperhead",
        group=group,  # seeds of the same setup, like "dqn-cartpole"
        name=name,
        config={"seed": seed, "n_steps": n_steps, **(config or {})},
    )

    # the training env, only the first reset is seeded
    env = make_env()
    obs, info = env.reset(seed=seed)
    episode_return, episode_length = 0.0, 0
    best_eval = float("-inf")

    # an env with render_mode="rgb_array", used to record a gif at every evaluation
    if make_video_env is not None:
        from coppercore.dqn.snake_render import log_episode_gif, record_episode

        video_env = make_video_env()

    for step in trange(1, n_steps + 1):
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
                np.savez(f"{run.dir}/best.npz", **getattr(agent, "params", {}))
                run.summary["best_eval_return"] = best_eval

                # record the best agent now, the gif is logged once after training
                if make_video_env is not None:
                    best_frames, stats = record_episode(
                        video_env,
                        lambda o: agent.act(o, greedy=True),
                        seed=10_000,
                        max_frames=5000,
                    )
                    best_caption = (
                        f"step={step} len={stats['length']} "
                        f"return={stats['return']:.1f}"
                    )

    if make_video_env is not None:
        log_episode_gif(
            run, best_frames, n_steps, key="eval/best_episode", caption=best_caption
        )

    # upload the best weights as a WandB artifact
    artifact = wandb.Artifact(f"agent-{run.id}", type="model")
    artifact.add_file(f"{run.dir}/best.npz")
    run.log_artifact(artifact)
    run.finish()
