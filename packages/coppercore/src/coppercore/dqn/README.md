# snake_env

A Snake environment for reinforcement learning. `SnakeEnv` uses Gymnasium-style `reset` and `step` methods.

## Usage

```python
import random

from coppercore.dqn.snake_env import SnakeEnv

env = SnakeEnv(width=10, height=10, seed=0)
obs, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = random.choice(env.action_space)  # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)

print(info["length"])
```

| Argument            | Default   | Description                                             |
| ------------------- | --------- | ------------------------------------------------------- |
| `width`, `height`   | `10`      | Board size. `width` must be at least 4.                 |
| `seed`              | `None`    | Seed for apple placement.                               |
| `max_steps_no_food` | `100`     | Steps without eating before the episode is truncated.   |
| `rewards`           | see below | Dict with a reward for every `Event`.                   |
| `obs_mode`          | `"state"` | `"state"` for an integer, any other value for a vector. |

## Rules

- The snake starts in the middle of the board with length 3, heading right.
- Actions are `LEFT = 0`, `RIGHT = 1`, `UP = 2` and `DOWN = 3`. Reversing is ignored, so the snake continues straight ahead.
- An episode is `terminated` when the snake dies or fills the board, and `truncated` after `max_steps_no_food` steps without eating.

| Event   | When                 | Default reward |
| ------- | -------------------- | -------------- |
| `MOVED` | Ordinary step        | `0.0`          |
| `ATE`   | Ate an apple         | `1.0`          |
| `DIED`  | Hit a wall or itself | `-1.0`         |
| `WON`   | Filled the board     | `1.0`          |

## Observations

**`obs_mode="state"`** returns an integer from 0 to 255, for tabular methods. Each bit is a yes/no feature, ordered left, right, up, down:

- Bits 0–3: a wall or body part is next to the head in that direction.
- Bits 4–7: the apple lies in that direction.

**Any other value** returns a flat `float32` vector of length `3 * height * width`, for neural networks. It holds one indicator channel each for body, head and apple.

## Rendering

`RenderedSnakeEnv` in `snake_render.py` takes the same arguments, plus `render_mode`:

- `None` prints the board to the terminal.
- `"rgb_array"` returns an image. Use it with `record_episode` and `log_episode_gif` to log episodes to WandB.
- `"human"` opens a window. Set `SDL_VIDEODRIVER` to `""` before importing `snake_render`, since it runs headless by default.

To watch a random agent play, run:

```bash
uv run python -m coppercore.dqn.run_random_snake
```
