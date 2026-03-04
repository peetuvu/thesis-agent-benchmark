import pytest

from snake import init_state, step


def test_move_right_shifts_tail():
    s = init_state(
        5, 5,
        snake=[(2, 2), (1, 2), (0, 2)],
        food={"apple": (4, 4), "big": (4, 3)},
        direction="R",
    )
    s2 = step(s, "R")
    assert s2["snake"] == [(3, 2), (2, 2), (1, 2)]


def test_eat_grows_and_scores():
    s = init_state(
        5, 5,
        snake=[(2, 2), (1, 2), (0, 2)],
        food={"apple": (3, 2), "big": (4, 4)},
        direction="R",
    )
    s2 = step(s, "R")
    assert s2["score"] == 1
    assert len(s2["snake"]) == 4  # grew by 1 immediately (kept tail)
    assert s2["snake"][0] == (3, 2)


def test_wall_collision_kills():
    s = init_state(
        3, 3,
        snake=[(2, 1), (1, 1), (0, 1)],
        food={"apple": (0, 0), "big": (0, 2)},
        direction="R",
        wrap=False,
    )
    s2 = step(s, "R")
    assert s2["alive"] is False


def test_wrap_keeps_alive_and_wraps_position():
    s = init_state(
        3, 3,
        snake=[(2, 1), (1, 1)],
        food={"apple": (0, 0), "big": (0, 2)},
        direction="R",
        wrap=True,
    )
    s2 = step(s, "R")
    assert s2["alive"] is True
    assert s2["snake"][0] == (0, 1)


def test_self_collision_kills_after_tail_handling():
    # This shape makes moving up collide into itself even after tail movement
    s = init_state(
        5, 5,
        snake=[(2, 2), (2, 3), (1, 3), (1, 2), (1, 1), (2, 1)],
        food={"apple": (4, 4), "big": (4, 3)},
        direction="L",
    )
    s2 = step(s, "U")
    assert s2["alive"] is False


def test_obstacle_collision_kills():
    s = init_state(
        5, 5,
        snake=[(2, 2), (1, 2)],
        food={"apple": (4, 4), "big": (4, 3)},
        direction="R",
        obstacles=[(3, 2)],
    )
    s2 = step(s, "R")
    assert s2["alive"] is False


def test_big_food_adds_two_growth_over_two_steps():
    s = init_state(
        5, 5,
        snake=[(2, 2), (1, 2), (0, 2)],
        food={"apple": (4, 4), "big": (3, 2)},
        direction="R",
    )
    s2 = step(s, "R")  # eat big -> +2 score, growth pending
    assert s2["score"] == 2
    assert len(s2["snake"]) == 4  # first growth unit applied immediately
    s3 = step(s2, "R")
    assert len(s3["snake"]) == 5  # second growth unit applied next step


def test_deterministic_respawn_respects_other_food_and_obstacles():
    s = init_state(
        4, 3,
        snake=[(1, 1), (0, 1)],
        food={"apple": (2, 1), "big": (0, 0)},
        direction="R",
        obstacles=[(1, 0)],
    )
    # eat apple at (2,1)
    s2 = step(s, "R")
    # occupied: snake (after move), obstacles, and big at (0,0)
    # scanning: (0,0) blocked by big, (1,0) obstacle, (2,0) should be first free
    assert s2["food"]["apple"] == (2, 0)
