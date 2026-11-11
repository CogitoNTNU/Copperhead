import numpy as np
import pytest
from coppercore.dqn.snake_env import Action, Board, Event, SnakeEnv, Tile
from coppercore.dqn.snake_render import RenderedSnakeEnv


def test_snake_env_default_no_movement_rule():
    env = SnakeEnv(width=10, height=10, seed=0)
    assert not env.movement_rule
    env.reset()
    assert not env.board.movement_rule

    # Snake starts at (5, 5), heading RIGHT (0, 1).
    # With movement_rule=False, on odd col 5 it can turn UP (which would be forbidden under movement_rule)
    obs, reward, terminated, truncated, info = env.step(Action.UP)
    assert env.board.head == (4, 5)


def test_movement_rule_is_action_allowed():
    rng = np.random.default_rng(0)
    board = Board(width=10, height=10, length=3, rng=rng, movement_rule=True)

    # Place snake head at various test positions
    # (row, col)
    # Even row, Even col -> (4, 4)
    board.body[0] = (4, 4)
    assert not board.is_action_allowed(Action.UP)  # col 4 is even -> up forbidden
    assert board.is_action_allowed(Action.DOWN)  # col 4 is even
    assert board.is_action_allowed(Action.LEFT)  # row 4 is even
    assert not board.is_action_allowed(Action.RIGHT)  # row 4 is even -> right forbidden

    # Even row, Odd col -> (4, 5)
    board.body[0] = (4, 5)
    assert board.is_action_allowed(Action.UP)  # col 5 is odd
    assert not board.is_action_allowed(Action.DOWN)  # col 5 is odd -> down forbidden
    assert board.is_action_allowed(Action.LEFT)  # row 4 is even
    assert not board.is_action_allowed(Action.RIGHT)  # row 4 is even -> right forbidden

    # Odd row, Even col -> (5, 4)
    board.body[0] = (5, 4)
    assert not board.is_action_allowed(Action.UP)  # col 4 is even -> up forbidden
    assert board.is_action_allowed(Action.DOWN)  # col 4 is even
    assert not board.is_action_allowed(Action.LEFT)  # row 5 is odd -> left forbidden
    assert board.is_action_allowed(Action.RIGHT)  # row 5 is odd

    # Odd row, Odd col -> (5, 5)
    board.body[0] = (5, 5)
    assert board.is_action_allowed(Action.UP)  # col 5 is odd
    assert not board.is_action_allowed(Action.DOWN)  # col 5 is odd -> down forbidden
    assert not board.is_action_allowed(Action.LEFT)  # row 5 is odd -> left forbidden
    assert board.is_action_allowed(Action.RIGHT)  # row 5 is odd


def test_snake_env_movement_rule_stepping():
    env = SnakeEnv(width=10, height=10, seed=0, movement_rule=True)
    assert env.movement_rule
    env.reset()
    assert env.board.movement_rule

    # Initial snake head is at (5, 5), body at (5, 4), (5, 3), direction is RIGHT (0, 1) since row 5 is odd
    assert env.board.head == (5, 5)
    assert env.board.direction == (0, 1)

    # At (5, 5) (odd row 5, odd col 5):
    # - vertical: only UP is allowed
    # - horizontal: only RIGHT is allowed

    # Attempting DOWN on odd column 5 should be ignored, continuing RIGHT
    env.step(Action.DOWN)
    assert env.board.head == (5, 6)

    # Now at (5, 6) (odd row 5, even col 6):
    # - vertical: only DOWN is allowed
    # - horizontal: only RIGHT is allowed

    # Attempting UP on even col 6 should be ignored, continuing RIGHT
    env.step(Action.UP)
    assert env.board.head == (5, 7)

    # Now at (5, 7) (odd row 5, odd col 7):
    # Turn UP (allowed on odd col 7)
    env.step(Action.UP)
    assert env.board.head == (4, 7)

    # Now at (4, 7) (even row 4, odd col 7):
    # - vertical: only UP is allowed
    # - horizontal: only LEFT is allowed

    # Attempting RIGHT on even row 4 should be ignored, continuing UP
    env.step(Action.RIGHT)
    assert env.board.head == (3, 7)

    # Now at (3, 7) (odd row 3, odd col 7):
    # Turn RIGHT on odd row 3
    env.step(Action.RIGHT)
    assert env.board.head == (3, 8)

    # Now at (3, 8) (odd row 3, even col 8):
    # Turn DOWN on even col 8
    env.step(Action.DOWN)
    assert env.board.head == (4, 8)

    # Now at (4, 8) (even row 4, even col 8):
    # Turn LEFT on even row 4
    env.step(Action.LEFT)
    assert env.board.head == (4, 7)


def test_square_one_one_reachability():
    # Verify that cell (1, 1) is fully traversable and can be reached from (2, 1) and (1, 0)
    rng = np.random.default_rng(0)
    board = Board(width=10, height=10, length=3, rng=rng, movement_rule=True)

    # At (2, 1): even row 2 (LEFT), odd col 1 (UP).
    # Moving UP enters (1, 1)
    board.body[0] = (2, 1)
    assert board.is_action_allowed(Action.UP)

    # At (1, 0): odd row 1 (RIGHT), even col 0 (DOWN).
    # Moving RIGHT enters (1, 1)
    board.body[0] = (1, 0)
    assert board.is_action_allowed(Action.RIGHT)

    # At (1, 1): odd row 1 (RIGHT), odd col 1 (UP).
    board.body[0] = (1, 1)
    assert board.is_action_allowed(Action.RIGHT)
    assert board.is_action_allowed(Action.UP)
    assert not board.is_action_allowed(Action.LEFT)
    assert not board.is_action_allowed(Action.DOWN)


def test_snake_env_even_height_initialization():
    # height=8 (head_y=4, even row)
    env = SnakeEnv(width=8, height=8, seed=0, movement_rule=True)
    env.reset()
    # On even row 4, snake should face LEFT (0, -1)
    assert env.board.direction == (0, -1)
    assert env.board.head[0] == 4
    # Stepping LEFT is allowed
    env.step(Action.LEFT)
    assert env.board.head[0] == 4


@pytest.mark.parametrize(
    "width,height",
    [(4, 4), (6, 6), (8, 6), (6, 8), (10, 10), (16, 18), (5, 5), (7, 7)],
)
def test_snake_env_board_size_initialization(width, height):
    for movement_rule in (True, False):
        env = SnakeEnv(width=width, height=height, seed=0, movement_rule=movement_rule)
        obs, info = env.reset()
        assert env.board.head in env.board.body
        for part in env.board.body:
            assert 0 <= part[0] < height
            assert 0 <= part[1] < width
            assert env.board.tiles[part] in (Tile.HEAD, Tile.BODY)


def test_rendered_snake_env_movement_rule():
    env = RenderedSnakeEnv(width=10, height=10, seed=0, movement_rule=True)
    assert env.movement_rule
    env.reset()
    assert env.board.movement_rule


def test_space_after_moves():
    env = SnakeEnv(width=10, height=10, seed=0, movement_rule=True)
    env.reset()
    free_area, tail_reachable = env._space_after_moves()
    assert free_area.shape == (len(Action),)
    assert tail_reachable.shape == (len(Action),)


def test_board_is_action_allowed_when_disabled():
    rng = np.random.default_rng(0)
    board = Board(width=10, height=10, length=3, rng=rng, movement_rule=False)
    for action in Action:
        assert board.is_action_allowed(action)


def test_board_observation_gradient_and_wall_padding():
    env = SnakeEnv(width=8, height=6, seed=42, obs_mode="board")
    obs, info = env.reset()

    # Board shape with padding should be (3, height + 2, width + 2)
    assert obs.shape == (3, 8, 10)
    assert env.obs_shape == (3, 8, 10)
    assert env.obs_dim == 3 * 8 * 10

    # Channel 0: Head
    head_channel = obs[0]
    assert head_channel.sum() == 1.0
    head_y, head_x = env.board.head
    assert head_channel[head_y + 1, head_x + 1] == 1.0

    # Channel 1: Danger/body gradient + wall padding
    danger_channel = obs[1]
    # Check 1-layer wall perimeter is 1.0
    assert np.all(danger_channel[0, :] == 1.0)
    assert np.all(danger_channel[-1, :] == 1.0)
    assert np.all(danger_channel[:, 0] == 1.0)
    assert np.all(danger_channel[:, -1] == 1.0)

    # Check snake gradient from head (1.0) down to tail (1/length)
    body = list(env.board.body)
    length = len(body)
    for i, (by, bx) in enumerate(body):
        expected_val = (length - i) / length
        actual_val = danger_channel[by + 1, bx + 1]
        assert np.isclose(actual_val, expected_val)

    # Channel 2: Apple
    apple_channel = obs[2]
    assert apple_channel.sum() == 1.0
    apple_coords = np.argwhere(env.board.tiles == Tile.APPLE)[0]
    assert apple_channel[apple_coords[0] + 1, apple_coords[1] + 1] == 1.0
