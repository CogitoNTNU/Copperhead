from coppercore.dqn.snake_env import SnakeEnv, Action
from coppercore.training.dqn_harness import evaluate, train
from coppercore.dqn.snake_render import RenderedSnakeEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm   

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
        snapshot_steps=(1_000, 50_000, 250_000),
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

        # Plotting heatmap:
        self.step_count = 0
        self.snapshot_steps = set(snapshot_steps)
        self.snapshots = {}
        

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

        self.step_count += 1
        if self.step_count in self.snapshot_steps:
            self.snapshots[self.step_count] = self.qtable.copy()

        # training requires a dict 
        return {
            "q_value": self.qtable[obs, action],
        }

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# Creates a heatmap of Qtable values:
def plot_qtable_heatmap(agent, max_states=25, save_path="qtable_heatmap.png"):
    # Gather (state, q_values) pairs from either a dict or a numpy Q-table
    if isinstance(agent.qtable, dict):
        items = list(agent.qtable.items())
    else:
        items = [(s, agent.qtable[s]) for s in range(agent.qtable.shape[0])]

    # Skip states never updated (all zeros), keep the ones with the strongest values
    items = [(s, q) for s, q in items if np.any(q != 0)]
    items.sort(key=lambda sq: np.max(np.abs(sq[1])), reverse=True)
    items = items[:max_states]
    if not items:
        print("Q-table is empty – train first.")
        return

    states = [str(s) if len(str(s)) <= 30 else str(s)[:27] + "..." for s, _ in items]
    values = np.array([q for _, q in items])

    try:
        actions = [Action(a).name for a in range(values.shape[1])]
    except Exception:
        actions = [f"Action {a}" for a in range(values.shape[1])]

    fig, ax = plt.subplots(figsize=(1.6 * len(actions) + 3, 0.45 * len(states) + 2))
    im = ax.imshow(values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(actions)), labels=actions)
    ax.set_yticks(range(len(states)), labels=states, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_title("Q-values per state and action", pad=30)

    # Write the value in each cell, bold for the action the agent would pick
    norm = im.norm
    for i in range(values.shape[0]):
        best = np.argmax(values[i])
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black",
                    fontweight="bold" if j == best else "normal")

    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, fraction=0.04)
    fig.tight_layout()

def plot_learning_progress(agent, max_states=15):
    snaps = dict(agent.snapshots)
    snaps["end"] = agent.qtable

    final = agent.qtable
    states = [s for s in range(final.shape[0]) if np.any(final[s] != 0)]
    states.sort(key=lambda s: np.max(final[s]) - np.min(final[s]), reverse=True)
    states = states[:max_states]

    vmin, vmax = final[states].min(), final[states].max()
    norm = TwoSlopeNorm(vmin=min(vmin, -0.01), vcenter=0, vmax=vmax)   # 0 = yellow

    fig, axes = plt.subplots(1, len(snaps), figsize=(4 * len(snaps), 0.4 * len(states) + 2),
                             sharey=True)
    actions = [Action(a).name for a in range(final.shape[1])]

    for ax, (step, table) in zip(axes, snaps.items()):
        im = ax.imshow(table[states], cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_title(f"Step {step}")
        ax.set_xticks(range(len(actions)), labels=actions, fontsize=8)
    axes[0].set_yticks(range(len(states)), labels=states, fontsize=8)

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08)

def make_env():
    env = SnakeEnv(width=10, height=10, seed=0)
    obs, info = env.reset()
    return env

def main():
    n_steps = 500_000
    learning_rate = 0.1
    start_epsilon = 1.0
    final_epsilon = 0.05
    epsilon_decay = (start_epsilon - final_epsilon) / (n_steps * 0.5)
    discount_factor = 0.99

    env = make_env()
    agent = QLearningAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        seed=42,
        n_states=env.n_states,
        n_actions=env.n_actions,
        discount_factor=discount_factor
   )

    train(
        agent=agent,
        make_env=make_env,
        n_steps=n_steps,
        make_video_env=lambda: RenderedSnakeEnv(
            width=10, height=10, max_steps_no_food=200, render_mode="rgb_array"
        ),
    )
    plot_qtable_heatmap(agent)
    plot_learning_progress(agent)
    plt.show()

    evaluate(agent=agent, make_env=make_env)


if __name__ == "__main__":
    main()