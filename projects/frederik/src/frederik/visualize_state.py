"""
Visualizes the snake state representations from coppercore.dqn.snake_env.

A hand-made board position is turned into
- the observation vector used by the DQN (obs_mode="obs")
- the 8-bit integer used by the Q-table (obs_mode="state")
and both are drawn next to the board.

Run: uv run python -m frederik.visualize_state [--out DIR] [--show]

Made with Claude.
"""

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle

from coppercore.dqn.snake_env import DIRS, Action, SnakeEnv, Tile

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]

# colors
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BODY = "#86b6ef"
HEAD = "#184f95"
APPLE = "#e34948"
WALL = "#52514e"
WINDOW = "#eb6834"

DANGER_CMAP = LinearSegmentedColormap.from_list("danger", [SURFACE, "#86b6ef", HEAD])
APPLE_CMAP = LinearSegmentedColormap.from_list("apple", [SURFACE, APPLE])
VECTOR_CMAP = LinearSegmentedColormap.from_list(
    "vector", ["#e34948", "#f0efec", "#2a78d6"]
)

ACTION_NAMES = {
    Action.LEFT: "left",
    Action.RIGHT: "right",
    Action.UP: "up",
    Action.DOWN: "down",
}

# body from head to tail as (row, col). The head moves left and the three tiles
# below it are a dead end, which the DQN features see and the 8-bit state does not.
SNAKE = [
    (1, 4), (1, 5), (1, 6), (1, 7), (2, 7), (3, 7), (3, 6), (3, 5),
    (3, 4), (3, 3), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (5, 3),
]  # fmt: skip
APPLE_POS = (0, 6)


def make_example_env(width: int = 10, height: int = 10, radius: int = 2) -> SnakeEnv:
    """Returns an env whose board is replaced by the example position."""
    env = SnakeEnv(
        width=width, height=height, seed=0, obs_mode="obs", obs_window_radius=radius
    )
    env.reset()

    board = env.board
    board.tiles[:] = Tile.EMPTY
    board.body = deque(SNAKE)
    for part in SNAKE:
        board.tiles[part] = Tile.BODY
    board.tiles[board.head] = Tile.HEAD
    board.tiles[APPLE_POS] = Tile.APPLE
    return env


def split_obs(env: SnakeEnv, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Splits the observation vector into its named parts, in the order of get_obs."""
    size = 2 * env.obs_window_radius + 1
    sizes = {
        "danger": size * size,
        "apple": size * size,
        "apple_dir": 2,
        "snake_dir": 2,
        "free_area": len(Action),
        "tail": len(Action),
        "length": 1,
    }
    parts, start = {}, 0
    for name, n in sizes.items():
        parts[name] = obs[start : start + n]
        start += n
    assert start == len(obs) == env.obs_dim

    parts["danger"] = parts["danger"].reshape(size, size)
    parts["apple"] = parts["apple"].reshape(size, size)
    return parts


def outside_mask(env: SnakeEnv) -> np.ndarray:
    """True for window tiles that lie outside the board."""
    radius = env.obs_window_radius
    head_y, head_x = env.board.head
    offsets = np.arange(-radius, radius + 1)
    rows, cols = head_y + offsets[:, None], head_x + offsets[None, :]
    return (rows < 0) | (rows >= env.height) | (cols < 0) | (cols >= env.width)


# drawing helpers


def draw_board(ax, env: SnakeEnv, show_window: bool = True, neighbours: bool = False):
    """Draws the board with snake, apple and optionally the observation window."""
    board, radius = env.board, env.obs_window_radius
    height, width = board.tiles.shape

    colors = {
        Tile.EMPTY: SURFACE,
        Tile.BODY: BODY,
        Tile.HEAD: HEAD,
        Tile.APPLE: APPLE,
    }
    image = np.array([[to_rgb(colors[Tile(t)]) for t in row] for row in board.tiles])
    ax.imshow(image, interpolation="nearest")

    # tile grid and board edge
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=1)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.tick_params(labelsize=8, labelcolor=MUTED)
    ax.add_patch(
        Rectangle((-0.5, -0.5), width, height, fill=False, ec=WALL, lw=3, zorder=3)
    )

    # snake path from head to tail
    ys, xs = zip(*board.body)
    ax.plot(xs, ys, color=HEAD, lw=1.5, alpha=0.6, zorder=2)
    head_y, head_x = board.head
    tail_y, tail_x = board.body[-1]
    ax.text(head_x, head_y, "H", ha="center", va="center", color="white", weight="bold")
    ax.text(tail_x, tail_y, "T", ha="center", va="center", color=HEAD, weight="bold")

    margin = 0.5
    if show_window:
        size = 2 * radius + 1
        ax.add_patch(
            Rectangle(
                (head_x - radius - 0.5, head_y - radius - 0.5),
                size,
                size,
                fill=False,
                ec=WINDOW,
                lw=2.5,
                ls="--",
                zorder=4,
                clip_on=False,
            )
        )
        # hatch the part of the window that lies outside the board
        for dy, row in enumerate(outside_mask(env)):
            for dx, out in enumerate(row):
                if out:
                    ax.add_patch(
                        Rectangle(
                            (head_x - radius + dx - 0.5, head_y - radius + dy - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="///",
                            ec=WALL,
                            lw=0,
                            clip_on=False,
                        )
                    )
        margin += radius

    if neighbours:
        labels = {Action.LEFT: "←", Action.RIGHT: "→", Action.UP: "↑", Action.DOWN: "↓"}
        for action, label in labels.items():
            dy, dx = DIRS[action]
            y, x = head_y + dy, head_x + dx
            ax.add_patch(
                Rectangle(
                    (x - 0.45, y - 0.45),
                    0.9,
                    0.9,
                    fill=False,
                    ec=WINDOW,
                    lw=2.5,
                    zorder=4,
                )
            )
            on_body = board.tiles[y, x] == Tile.BODY
            ax.text(x, y, label, ha="center", va="center", fontsize=13, weight="bold",
                    color="white" if on_body else WINDOW, zorder=6)  # fmt: skip

        # dashed line from head to apple
        apple_y, apple_x = np.argwhere(board.tiles == Tile.APPLE)[0]
        ax.add_patch(
            FancyArrowPatch(
                (head_x, head_y),
                (apple_x, apple_y),
                arrowstyle="-|>",
                mutation_scale=14,
                color=APPLE,
                ls="--",
                lw=1.5,
                shrinkA=8,
                shrinkB=8,
                zorder=5,
            )
        )

    ax.set_xlim(-margin, width - 1 + margin)
    ax.set_ylim(height - 1 + margin, -margin)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_window(ax, values, mask, title, cmap, fmt="{:.2f}", outside_color=WALL):
    """Draws a head-centered window with its values written in each tile."""
    size = values.shape[0]
    radius = size // 2
    ax.imshow(values, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    for y in range(size):
        for x in range(size):
            v = values[y, x]
            dark = v > 0.55
            if mask[y, x]:
                # tiles outside the board get their own color so they read as wall
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=outside_color))
                dark = outside_color == WALL
                ax.text(x, y + 0.3, "outside", ha="center", va="center", fontsize=7,
                        color="white" if dark else INK_2)  # fmt: skip
            ax.text(
                x,
                y,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if dark else INK,
            )

    # mark the head tile in the center
    ax.add_patch(
        Rectangle((radius - 0.5, radius - 0.5), 1, 1, fill=False, ec=WINDOW, lw=2.5)
    )

    offsets = [f"{o:+d}" if o else "0" for o in range(-radius, radius + 1)]
    ax.set_xticks(range(size), offsets)
    ax.set_yticks(range(size), offsets)
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color=GRID, lw=1)
    ax.tick_params(which="both", length=0, labelsize=8, labelcolor=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, color=INK, loc="left")


def draw_directions(ax, apple_dir, snake_dir):
    """Draws the apple direction and the snake direction as arrows."""
    items = [
        (0.0, apple_dir, APPLE, "direction to apple"),
        (3.2, snake_dir, HEAD, "snake direction"),
    ]
    for cx, (dy, dx), color, label in items:
        ax.add_patch(plt.Circle((cx, 0), 1.1, fill=False, ec=GRID, lw=1.5))
        ax.plot([cx - 1.1, cx + 1.1], [0, 0], color=GRID, lw=1)
        ax.plot([cx, cx], [-1.1, 1.1], color=GRID, lw=1)
        # rows grow downwards on the board, so dy is flipped for the plot
        ax.add_patch(
            FancyArrowPatch(
                (cx, 0),
                (cx + 0.9 * dx, -0.9 * dy),
                arrowstyle="-|>",
                mutation_scale=18,
                color=color,
                lw=2.5,
            )
        )
        ax.text(cx, -1.45, label, ha="center", va="top", fontsize=10, color=INK)
        ax.text(
            cx,
            -1.9,
            f"(dy, dx) = ({dy:+.0f}, {dx:+.0f})",
            ha="center",
            va="top",
            fontsize=9,
            color=INK_2,
        )
    ax.set_xlim(-1.5, 4.7)
    ax.set_ylim(-2.4, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Directions (sign)", fontsize=11, color=INK, loc="left")


def draw_moves(ax, env: SnakeEnv, free_area, tail):
    """Draws free area and tail reachability for each action as bars."""
    actions = list(Action)
    # the action opposite to the current direction is ignored by the env
    dy, dx = env.board.direction
    reverse = [a for a in actions if DIRS[a] == (-dy, -dx)]

    x = np.arange(len(actions))
    ax.bar(x, free_area, width=0.55, color=BODY, edgecolor=HEAD, lw=0)
    for i, action in enumerate(actions):
        area = free_area[i]
        ax.text(i, area + 0.03, f"{area:.2f}", ha="center", va="bottom", fontsize=10,
                color=INK)  # fmt: skip
        reachable = bool(tail[i])
        ax.text(
            i,
            1.22,
            f"tail reachable: {'yes' if reachable else 'no'}",
            ha="center",
            va="center",
            fontsize=9,
            color=INK if reachable else APPLE,
            weight="normal" if reachable else "bold",
        )
        if area == 0 or not reachable:
            ax.text(i, area + 0.15, "dead end", ha="center", va="bottom", fontsize=10,
                    color=APPLE, weight="bold")  # fmt: skip
        if action in reverse:
            ax.text(i, -0.1, "(reversing is ignored = straight ahead)", ha="center", va="top",
                    fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())  # fmt: skip

    ax.set_xticks(x, [ACTION_NAMES[a] for a in actions])
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, labelcolor=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(
        "Per action: free area after the move (fraction of all free tiles)",
        fontsize=11,
        color=INK,
        loc="left",
    )


def draw_length(ax, env: SnakeEnv, length):
    ax.axis("off")
    ax.set_title("Length", fontsize=11, color=INK, loc="left")
    n = len(env.board.body)
    ax.text(
        0.5, 0.58, f"{length:.2f}", ha="center", va="center", fontsize=34, color=INK
    )
    ax.text(
        0.5,
        0.25,
        f"{n} segments / {env.width * env.height} tiles",
        ha="center",
        va="center",
        fontsize=10,
        color=INK_2,
    )


def draw_vector(ax, env: SnakeEnv, obs):
    """Draws the whole observation vector as a strip of cells, grouped by feature."""
    window = (2 * env.obs_window_radius + 1) ** 2
    groups = [
        ("danger window", window),
        ("apple window", window),
        ("apple\ndir.", 2),
        ("snake\ndir.", 2),
        ("free\narea", 4),
        ("tail\nreachable", 4),
        ("length", 1),
    ]

    for i, v in enumerate(obs):
        ax.add_patch(
            Rectangle((i, 0), 1, 1, fc=VECTOR_CMAP((v + 1) / 2), ec=SURFACE, lw=1)
        )
        if v != 0:
            text = (
                f"{v:.0f}" if v == int(v) else f"{v:.2f}".lstrip("0").replace("-0", "-")
            )
            ax.text(i + 0.5, 0.5, text, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else INK, rotation=90)  # fmt: skip

    start = 0
    for name, n in groups:
        end = start + n
        ax.plot([start + 0.1, end - 0.1], [1.25, 1.25], color=INK_2, lw=1.2)
        ax.plot([start + 0.1] * 2, [1.15, 1.25], color=INK_2, lw=1.2)
        ax.plot([end - 0.1] * 2, [1.15, 1.25], color=INK_2, lw=1.2)
        ax.text((start + end) / 2, 1.4, f"{name}\n[{start}:{end}]" if n > 4 else name,
                ha="center", va="top", fontsize=8.5, color=INK)  # fmt: skip
        start = end

    ax.set_xlim(-0.2, len(obs) + 0.2)
    ax.set_ylim(2.5, -0.1)
    ax.axis("off")
    ax.set_title(
        f"The full observation vector fed to the network ({len(obs)} numbers)   "
        "red = −1 · gray = 0 · blue = +1",
        fontsize=11,
        color=INK,
        loc="left",
    )


# figures


def dqn_figure(env: SnakeEnv):
    obs = env.get_obs()
    parts = split_obs(env, obs)
    mask = outside_mask(env)
    size = 2 * env.obs_window_radius + 1

    fig = plt.figure(figsize=(16, 11), layout="constrained", facecolor=SURFACE)
    grid = fig.add_gridspec(3, 4, height_ratios=[3.2, 2, 1.1])

    ax = fig.add_subplot(grid[0, 0:2])
    draw_board(ax, env, show_window=True)
    ax.set_title(
        f"The board: the head (H) moves left, dashed = {size}×{size} window around the head",
        fontsize=11,
        color=INK,
        loc="left",
    )

    draw_window(
        fig.add_subplot(grid[0, 2]),
        parts["danger"],
        mask,
        "Danger window\nwall = 1, segment = fraction of steps until it moves away",
        DANGER_CMAP,
    )
    draw_window(
        fig.add_subplot(grid[0, 3]),
        parts["apple"],
        mask,
        "Apple window\n1 where there is an apple",
        APPLE_CMAP,
        fmt="{:.0f}",
        outside_color=GRID,
    )

    draw_directions(fig.add_subplot(grid[1, 0]), parts["apple_dir"], parts["snake_dir"])
    draw_moves(fig.add_subplot(grid[1, 1:3]), env, parts["free_area"], parts["tail"])
    draw_length(fig.add_subplot(grid[1, 3]), env, parts["length"][0])
    draw_vector(fig.add_subplot(grid[2, :]), env, obs)

    fig.suptitle(
        'DQN observation (obs_mode="obs")',
        fontsize=16,
        color=INK,
        x=0.01,
        ha="left",
        weight="bold",
    )
    return fig


def qtable_figure(env: SnakeEnv):
    state = env.get_state()
    bits = [(state >> i) & 1 for i in range(8)]
    names = [ACTION_NAMES[a] for a in Action]

    fig = plt.figure(figsize=(16, 6.5), layout="constrained", facecolor=SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.9])

    ax = fig.add_subplot(grid[0, 0])
    draw_board(ax, env, show_window=False, neighbours=True)
    ax.set_title(
        "The board: the four tiles next to the head and the direction to the apple",
        fontsize=11,
        color=INK,
        loc="left",
    )

    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")
    gap = 0.6  # space between the danger and apple groups
    for i, bit in enumerate(bits):
        x = i + (gap if i >= 4 else 0)
        ax.add_patch(
            Rectangle((x, 0), 0.9, 0.9, fc=HEAD if bit else SURFACE, ec=HEAD, lw=1.5)
        )
        ax.text(x + 0.45, 0.45, str(bit), ha="center", va="center", fontsize=20,
                color="white" if bit else INK)  # fmt: skip
        ax.text(x + 0.45, -0.15, names[i % 4], ha="center", va="bottom", fontsize=10,
                color=INK)  # fmt: skip
        ax.text(
            x + 0.45, 1.05, f"bit {i}", ha="center", va="top", fontsize=9, color=MUTED
        )
        ax.text(x + 0.45, 1.35, f"× {1 << i}", ha="center", va="top", fontsize=10,
                color=INK if bit else MUTED)  # fmt: skip

    ax.text(1.95, -0.55, "danger (body or wall)", ha="center", va="bottom",
            fontsize=12, color=INK, weight="bold")  # fmt: skip
    ax.text(6.55, -0.55, "apple in that direction", ha="center", va="bottom",
            fontsize=12, color=INK, weight="bold")  # fmt: skip

    terms = " + ".join(str(1 << i) for i, b in enumerate(bits) if b) or "0"
    ax.text(0, 2.0, f"state = {terms} = {state}", fontsize=20, color=INK, va="center")
    ax.text(
        0,
        2.55,
        "8 yes/no questions → one integer 0–255 = one row in the Q-table",
        fontsize=11,
        color=INK_2,
        va="center",
    )
    ax.text(
        0,
        3.05,
        "Note: moving down leads into a dead end, but that bit is 0 – the local\n"
        "state only sees one tile ahead. The DQN observation catches this.",
        fontsize=10,
        color=APPLE,
        va="center",
    )
    ax.set_xlim(-0.3, 8.9)
    ax.set_ylim(3.7, -0.8)

    fig.suptitle(
        'Q-table state (obs_mode="state")',
        fontsize=16,
        color=INK,
        x=0.01,
        ha="left",
        weight="bold",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("state"))
    parser.add_argument("--show", action="store_true", help="open the figures")
    args = parser.parse_args()

    env = make_example_env()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, fig in [
        ("dqn_obs", dqn_figure(env)),
        ("qtable_state", qtable_figure(env)),
    ]:
        path = args.out / f"{name}.png"
        fig.savefig(path, dpi=160, facecolor=SURFACE)
        print(f"saved {path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
