import numpy as np
from src.dqn.snake_env import SnakeEnv, Action

env = SnakeEnv(width=10, height=10, seed=0)
obs, info = env.reset()
env.render()

for i in range(100):
    action = Action(np.random.randint(0, 4))
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n {action.name}:")
    env.render()

    if terminated or truncated:
        break

print(f"reward={reward}  terminated={terminated}  truncated={truncated}  info={info}")
print(f"obs.shape={obs.shape}")
