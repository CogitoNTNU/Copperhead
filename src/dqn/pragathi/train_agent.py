import gymnasium as gym
from .cart_pole import CartPoleAgent
from coppercore.dqn.snake_env import SnakeEnv, Action
from coppercore.training.dqn_harness import evaluate, train
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def save_gif(agent, path="cartpole.gif", n_episodes=1, fps=30):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        frames.append(env.render())
        done, total = False, 0
        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            total += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: return {total}")
    env.close()

    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"Saved {len(frames)} frames to {path}")

def fmt(s):
    if isinstance(s, tuple):
        return "(" + ", ".join(f"{float(x):.2f}" for x in s) + ")"
    return str(s)

def plot_qtable_heatmap(agent, max_states=25, save_path="qtable_heatmap.png",
                        action_names=None, title="Q-values per state and action"):
    # Find the Q-table, whatever the agent calls it
    qtable = getattr(agent, "qtable", None)
    if qtable is None:
        qtable = getattr(agent, "q_values")

    # Gather (state, q_values) pairs from either a dict or a numpy Q-table
    if isinstance(qtable, dict):
        items = list(qtable.items())
    else:
        items = [(s, qtable[s]) for s in range(qtable.shape[0])]

    # Skip states never updated (all zeros), keep the ones with the strongest values
    items = [(s, q) for s, q in items if np.any(q != 0)]
    items.sort(key=lambda sq: np.max(np.abs(sq[1])), reverse=True)
    items = items[:max_states]
    if not items:
        print("Q-table is empty – train first.")
        return

    states = [fmt(s) for s, _ in items]
    values = np.array([q for _, q in items])

    if action_names is None:
        action_names = [f"Action {a}" for a in range(values.shape[1])]

    fig, ax = plt.subplots(figsize=(1.6 * len(action_names) + 3, 0.45 * len(states) + 2))
    im = ax.imshow(values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(action_names)), labels=action_names)
    ax.set_yticks(range(len(states)), labels=states, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_title(title, pad=30)

    norm = im.norm
    for i in range(values.shape[0]):
        best = np.argmax(values[i])
        for j in range(values.shape[1]):
            v = values[i, j]
            dark = norm(v) < 0.15 or norm(v) > 0.85
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if dark else "black",
                    fontweight="bold" if j == best else "normal")

    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.04)
    fig.tight_layout()

def make_env():
    return gym.make("CartPole-v1")

def main():
    learning_rate = 0.1
    n_episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = CartPoleAgent(
        env=make_env(),
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    train(
        agent=agent,
        make_env=make_env,
        n_steps=42,
    )
    save_gif(agent, path="cartpole_random.gif", n_episodes=3)
    plot_qtable_heatmap(agent, action_names=["Push LEFT", "Push RIGHT"],
                        title="CartPole Q-values (random agent)")
    plt.show()
    evaluate(agent=agent, make_env=make_env)

if __name__ == "__main__":
    main()
