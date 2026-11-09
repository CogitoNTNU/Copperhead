import numpy as np
import numpy.typing as npt
from enum import Enum, IntEnum, auto
from collections import deque


class Tile(IntEnum):
    WALL = -1
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
        obs_window_radius: int = 2,
    ):
        self.width = width
        self.height = height
        self.snake_length = 3
        self.max_steps_no_food = max_steps_no_food
        self.rewards = rewards or {
            Event.MOVED: -0.01,
            Event.ATE: 2.0,
            Event.DIED: -1.0,
            Event.WON: 1.0,
        }
        self.rng = np.random.default_rng(seed)
        self.obs_mode = obs_mode
        self.obs_window_radius = obs_window_radius
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

    @property
    def obs_dim(self) -> int:
        """Length of the vector returned by get_obs."""
        window_size = (2 * self.obs_window_radius + 1) ** 2

        # danger and apple windows, apple and snake direction, area and tail per action, length
        return 2 * window_size + 2 + 2 + 2 * len(Action) + 1

    def _free_components(self) -> tuple[np.ndarray, list[int]]:
        """Labels connected free regions (empty, apple and the tail, which moves away)."""
        tiles = self.board.tiles
        free = (tiles == Tile.EMPTY) | (tiles == Tile.APPLE)
        free[self.board.body[-1]] = True

        h, w = tiles.shape
        labels = np.zeros((h, w), dtype=np.int32)  # 0 = not free
        sizes = [0]
        for sy, sx in np.argwhere(free):
            if labels[sy, sx]:
                continue
            label = len(sizes)
            labels[sy, sx] = label
            stack, size = [(sy, sx)], 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in DIRS_ARRAY:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and free[ny, nx]
                        and not labels[ny, nx]
                    ):
                        labels[ny, nx] = label
                        stack.append((ny, nx))
            sizes.append(size)
        return labels, sizes

    def get_obs(self) -> np.ndarray:
        """
        Head-centered window plus global features that help avoid self-trapping:
        - window danger: walls = 1, body parts = fraction of steps until they move away
        - window apples
        - direction to apple and snake direction
        - per action: reachable free area after the move (fraction of all free cells)
        - per action: whether the tail is reachable after the move
        - snake length as a fraction of the board
        """
        snake_length = len(self.board.body)

        # outside the board counts as wall
        danger_window = self._window(self._danger_map(), outside=1.0)
        apple_window = self._window(self.board.tiles == Tile.APPLE, outside=False)

        free_area, tail_reachable = self._space_after_moves()

        return np.concatenate(
            [
                danger_window.ravel(),
                apple_window.ravel(),
                self._apple_direction(),
                self.board.direction,
                free_area,
                tail_reachable,
                [snake_length / (self.width * self.height)],
            ]
        ).astype(np.float32)

    def _window(self, grid: npt.NDArray, outside: float) -> npt.NDArray:
        """Returns the square around the head, with tiles outside the board set to outside."""
        radius = self.obs_window_radius
        size = 2 * radius + 1
        head_y, head_x = self.board.head

        # after padding, the window starting at the head has the head in its center
        padded = np.pad(grid, radius, constant_values=outside)
        return padded[head_y : head_y + size, head_x : head_x + size]

    def _danger_map(self) -> npt.NDArray[np.float32]:
        """Returns the board with each body part as the fraction of steps until it moves away."""
        body = np.array(self.board.body)
        length = len(body)

        # body part i (head = 0) leaves its tile after length - i steps
        danger = np.zeros(self.board.tiles.shape, dtype=np.float32)
        danger[body[:, 0], body[:, 1]] = (length - np.arange(length)) / length
        return danger

    def _apple_direction(self) -> npt.NDArray:
        """Returns the sign of (dy, dx) from the head to the apple."""
        apples = np.argwhere(self.board.tiles == Tile.APPLE)

        if len(apples) == 0:  # board full, no apple
            return np.zeros(2)

        return np.sign(apples[0] - np.array(self.board.head))

    def _space_after_moves(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Returns, per action, the free area after the move and if the tail is reachable."""
        labels, sizes = self._free_components()
        n_free = max(1, sum(sizes))
        tail_label = labels[self.board.body[-1]]

        free_area = np.zeros(len(Action), dtype=np.float32)
        tail_reachable = np.zeros(len(Action), dtype=np.float32)

        for action in Action:
            new_pos = self.board.next_pos(action)

            # moving into a wall or the body leaves no space
            if not self.board.in_bounds(new_pos) or labels[new_pos] == 0:
                continue

            free_area[action] = sizes[labels[new_pos]] / n_free
            tail_reachable[action] = labels[new_pos] == tail_label

        return free_area, tail_reachable

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

        info = {"score": len(self.board.body)}

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
