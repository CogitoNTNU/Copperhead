import os

# headless rendering, so it also runs on servers and clusters without a display
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from typing import Callable

import numpy as np
import pygame
import wandb

from coppercore.dqn.snake_env import SnakeEnv, Tile


class TileRenderer:
    """Draws a (height, width) grid of Tile values to an RGB image."""

    BACKGROUND = (18, 18, 24)
    GRID = (32, 32, 42)
    TEXT = (220, 220, 220)
    COLORS = {
        Tile.APPLE: (235, 80, 80),
        Tile.BODY: (80, 200, 120),
        Tile.HEAD: (170, 255, 190),
    }
    DEAD_HEAD = (255, 60, 60)

    def __init__(
        self, height: int, width: int, cell_size: int = 24, hud_height: int = 24
    ):
        self.cell_size = cell_size
        self.hud_height = hud_height

        # pygame uses (width, height), the hud is a text bar above the board
        self.size = (width * cell_size, height * cell_size + hud_height)
        self.surface = pygame.Surface(self.size)  # off-screen surface

        pygame.font.init()
        self.font = pygame.font.SysFont(None, hud_height)
        self.window = None
        self.clock = None

    def draw(
        self, tiles: np.ndarray, text: str = "", dead: bool = False
    ) -> pygame.Surface:
        """Draws the board and the hud text to the off-screen surface."""
        surface, cell, top = self.surface, self.cell_size, self.hud_height
        height, width = tiles.shape
        surface.fill(self.BACKGROUND)

        # grid lines
        for x in range(width + 1):
            pygame.draw.line(
                surface, self.GRID, (x * cell, top), (x * cell, top + height * cell)
            )
        for y in range(height + 1):
            pygame.draw.line(
                surface, self.GRID, (0, top + y * cell), (width * cell, top + y * cell)
            )

        # tiles are indexed [row, column] = [y, x]
        for (y, x), tile in np.ndenumerate(tiles):
            if tile == Tile.EMPTY:
                continue

            if dead and tile == Tile.HEAD:
                color = self.DEAD_HEAD
            else:
                color = self.COLORS[Tile(tile)]

            # apples are drawn a bit smaller than the snake
            pad = 3 if tile == Tile.APPLE else 1
            rect = (
                x * cell + pad,
                top + y * cell + pad,
                cell - 2 * pad,
                cell - 2 * pad,
            )
            pygame.draw.rect(surface, color, rect)

        if text:
            surface.blit(self.font.render(text, True, self.TEXT), (6, 4))

        return surface

    def to_array(self) -> np.ndarray:
        """Returns the surface as a (height, width, 3) uint8 array."""
        # surfarray returns (width, height, 3), so swap the first two axes
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def show(self, fps: int = 10):
        """Shows the surface in a window, opened on the first call."""
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode(self.size)
            self.clock = pygame.time.Clock()

        pygame.event.pump()
        self.window.blit(self.surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        """Closes the window if it is open."""
        if self.window is not None:
            pygame.display.quit()
            self.window = None


class RenderedSnakeEnv(SnakeEnv):
    """SnakeEnv with render_mode set to None, "rgb_array" or "human"."""

    def __init__(self, *args, render_mode: str = None, cell_size: int = 24, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_mode = render_mode

        if render_mode is not None:
            self.renderer = TileRenderer(self.height, self.width, cell_size)
        else:
            self.renderer = None

    def render(self, text: str = "", dead: bool = False) -> np.ndarray | None:
        """Renders the board. Returns an RGB array if render_mode is "rgb_array"."""
        if self.render_mode is None:
            return super().render()  # prints the board to the terminal

        self.renderer.draw(self.board.tiles, text, dead)

        if self.render_mode == "rgb_array":
            return self.renderer.to_array()

        self.renderer.show()
        return None

    def close(self):
        """Closes the renderer."""
        if self.renderer is not None:
            self.renderer.close()


def record_episode(
    env: RenderedSnakeEnv,
    policy: Callable,
    seed: int = None,
    max_frames: int = 500,
    hold_last: int = 6,
) -> tuple[np.ndarray, dict]:
    """Plays one episode with policy(obs) -> action.
    Returns the frames as a (T, height, width, 3) uint8 array and a dict of stats.
    """
    obs, info = env.reset(seed=seed)
    frames = [env.render(f"len {len(env.board.body)}  step 0")]
    episode_return, step = 0.0, 0
    info = {"length": len(env.board.body)}

    for step in range(1, max_frames + 1):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        episode_return += reward

        # the episode is won if there is no apple left on the board
        won = terminated and not np.any(env.board.tiles == Tile.APPLE)
        dead = terminated and not won
        frames.append(env.render(f"len {info['length']}  step {step}", dead=dead))

        if terminated or truncated:
            break

    # repeat the last frame, so the end of the episode is shown a bit longer
    frames += [frames[-1]] * hold_last

    stats = {"return": episode_return, "length": info["length"], "steps": step}
    return np.stack(frames), stats


def log_episode_gif(
    run,
    frames: np.ndarray,
    step: int,
    key: str = "eval/episode",
    fps: int = 10,
    caption: str = None,
):
    """Logs the frames as a gif to WandB."""
    # WandB expects (T, C, H, W), while the frames are (T, H, W, C)
    video = np.transpose(frames, (0, 3, 1, 2))
    run.log(
        {key: wandb.Video(video, fps=fps, format="gif", caption=caption)}, step=step
    )


if __name__ == "__main__":
    # example with tabular Q-learning (obs_mode="state" gives an integer 0-255)
    eval_env = RenderedSnakeEnv(width=10, height=10, render_mode="rgb_array")
    q_table = np.zeros((eval_env.n_states, eval_env.n_actions))  # your trained Q-table

    def greedy(state: int) -> int:
        return int(np.argmax(q_table[state]))

    # remove mode="offline" to log online
    run = wandb.init(project="copperhead-rl", entity="Copperhead", mode="offline")

    for step in range(0, 5001, 1000):
        # ... train on a separate env without rendering ...
        frames, stats = record_episode(eval_env, greedy, seed=0)
        caption = f"len={stats['length']} return={stats['return']:.1f}"
        log_episode_gif(run, frames, step, caption=caption)
        run.log({f"eval/{k}": v for k, v in stats.items()}, step=step)

    run.finish()
