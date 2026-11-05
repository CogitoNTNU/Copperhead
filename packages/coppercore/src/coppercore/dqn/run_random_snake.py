import os
import time

# snake_render defaults to the headless "dummy" driver, empty lets SDL pick a real one
os.environ["SDL_VIDEODRIVER"] = ""

import numpy as np
from coppercore.dqn.snake_env import Action
from coppercore.dqn.snake_render import RenderedSnakeEnv

FPS = 2

env = RenderedSnakeEnv(width=10, height=10, seed=0, render_mode="human")
obs, info = env.reset()
env.render("start")

for i in range(100):
    action = np.random.randint(0, 4)
    obs, reward, terminated, truncated, info = env.step(action)

    env.render(f"{Action(action).name}  len {info['length']}", dead=terminated)

    if terminated or truncated:
        break

    time.sleep(1 / FPS)

time.sleep(1)  # keep the last frame on screen before the window closes
env.close()

print(f"reward={reward}  terminated={terminated}  truncated={truncated}  info={info}")
