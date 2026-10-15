import numpy as np
import numpy.typing as npt
from enum import Enum, IntEnum, auto
from collections import deque


class Tile(IntEnum):
    EMPTY = 0
    APPLE = 1
    BODY = 2
    HEAD = 3


class Event(Enum):
    MOVED = auto()
    ATE = auto()
    DIED = auto()
    WON = auto()


class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


DIRS: dict[Action, tuple[int, int]] = {
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
}

DIRS_ARRAY = np.array(list(DIRS.values()))
# possible states in the Q-table
N_STATES = 2**8  # 4 danger bits + 4 apple bits


class Board:
    def __init__(self, width: int, height: int, length: int, rng: np.random.Generator):
        assert width // 2 >= length - 1, "the board is too small for the snake"

        self.width = width
        self.height = height
        self.length = length
        self.rng = rng

        self.tiles = np.full((self.height, self.width), Tile.EMPTY)

        # initilize snake with given length as a deque
        head_y, head_x = height // 2, width // 2
        self.body = deque((head_y, head_x - i) for i in range(self.length))

        for part in list(self.body):
            self.tiles[part] = Tile.BODY

        self.tiles[self.head] = Tile.HEAD

        # spawn first apple
        self.spawn_apple()

    @property
    def head(self) -> tuple[int, int]:
        """Returns the position of the snake's head."""
        return self.body[0]

    @property
    def direction(self) -> tuple[int, int]:
        """Returns the direction the snake is moving in as (dy, dx)."""
        (head_y, head_x), (neck_y, neck_x) = self.body[0], self.body[1]
        return (head_y - neck_y, head_x - neck_x)

    def spawn_apple(self) -> bool:
        """Places an apple on a random empty tile. Returns False if the board is full."""
        empty = np.flatnonzero(self.tiles == Tile.EMPTY)

        if len(empty) == 0:
            return False

        self.tiles.flat[self.rng.choice(empty)] = Tile.APPLE
        return True

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        """Checks if the given position is inside the board."""
        pos_y, pos_x = pos

        if pos_y < 0 or pos_y > self.height - 1:
            return False
        elif pos_x < 0 or pos_x > self.width - 1:
            return False

        return True

    def get_tile_neighbours(
        self, tile: Tile, outside: bool = False
    ) -> npt.NDArray[np.bool_]:
        """Check whether a tile type neighbours the head (up, down, left, right)."""
        neighbours = np.array(self.head) + DIRS_ARRAY
        xn, yn = neighbours[:, 0], neighbours[:, 1]

        # mask for neighbours that lie on the board
        inside = (xn >= 0) & (xn < self.height) & (yn >= 0) & (yn < self.width)

        result = np.full(4, outside, dtype=bool)
        result[inside] = self.tiles[xn[inside], yn[inside]] == tile
        return result

    def next_pos(self, action: Action) -> tuple[int, int]:
        """Returns the next head position for the given action. Reversing is ignored."""
        dy, dx = DIRS[action]

        ry, rx = self.direction

        # prevent illegal turn
        if (dy, dx) == (-ry, -rx):
            dy, dx = ry, rx

        return (self.head[0] + dy, self.head[1] + dx)

    def move(self, action: Action) -> Event:
        """Moves the snake one step and returns what happened."""
        new_pos = self.next_pos(action)

        if not self.in_bounds(new_pos):
            return Event.DIED

        ate = self.tiles[new_pos] == Tile.APPLE
        if not ate:
            tail = self.body.pop()
            self.tiles[tail] = Tile.EMPTY

        if self.tiles[new_pos] == Tile.BODY:
            return Event.DIED

        self.tiles[self.head] = Tile.BODY
        self.body.appendleft(new_pos)
        self.tiles[new_pos] = Tile.HEAD

        if ate:
            # won if board is full
            if not self.spawn_apple():
                return Event.WON
            return Event.ATE
        return Event.MOVED


class SnakeEnv:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int = None,
        max_steps_no_food: int = 100,
        rewards=None,
        obs_mode: str = "state",
    ):
        self.width = width
        self.height = height
        self.snake_length = 3
        self.max_steps_no_food = max_steps_no_food
        self.rewards = rewards or {
            Event.MOVED: 0.0,
            Event.ATE: 1.0,
            Event.DIED: -1.0,
            Event.WON: 1.0,
        }
        self.rng = np.random.default_rng(seed)
        self.obs_mode = obs_mode
        self.board = None

    @property
    def n_actions(self) -> int:
        """Returns the length of the action space."""
        return len(Action)

    @property
    def n_states(self) -> int:
        """Returns the length of the action space."""
        return N_STATES

    @property
    def action_space(self) -> tuple[Action, ...]:
        """Returns the action space."""
        return tuple(Action)

    def observe(self):
        return self.get_state() if self.obs_mode == "state" else self.get_obs()

    def get_state(self) -> int:
        # danger in each direction, including walls
        dangers_nearby = self.board.get_tile_neighbours(Tile.BODY, outside=True)

        # finding the direction of the apple
        head = np.array(self.board.head)
        apples = np.argwhere(self.board.tiles == Tile.APPLE)

        if len(apples) == 0:  # board full, no apple
            apple_dir = np.zeros(4, dtype=bool)
        else:
            diff = apples[0] - head  # vector from head to apple
            apple_dir = (DIRS_ARRAY @ diff) > 0  # True if apple in same direction

        # 8 bools to an integer 0-255
        bits = np.concatenate([dangers_nearby, apple_dir])
        return int(bits @ (1 << np.arange(bits.size)))

    def get_obs(self) -> np.ndarray:
        """Returns the board as a flat float vector for the network."""
        tiles = self.board.tiles
        # divide board in 3 channels, with 1 as indicator, and flatten to one float vector
        return (
            np.stack([tiles == Tile.BODY, tiles == Tile.HEAD, tiles == Tile.APPLE])
            .astype(np.float32)
            .ravel()
        )

    def step(self, action: int) -> tuple[tuple[np.ndarray], float, bool, bool, dict]:
        """Takes one action and returns (obs, reward, terminated, truncated, info)."""
        event = self.board.move(Action(action))

        terminated = event in (Event.DIED, Event.WON)

        if event == Event.ATE:
            self.steps_no_food = 0
        else:
            self.steps_no_food += 1

        truncated = self.steps_no_food >= self.max_steps_no_food
        next_obs = self.observe()
        reward = self.rewards[event]

        info = {"length": len(self.board.body)}

        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None) -> tuple[tuple, dict]:
        """Resets the environment with, optionally, a new seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board = Board(self.width, self.height, self.snake_length, self.rng)
        self.steps_no_food = 0

        return self.observe(), {}

    def render(self):
        """Prints the board to the terminal."""
        print(self.board.tiles)
