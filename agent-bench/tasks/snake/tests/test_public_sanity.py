import pytest

from snake import init_state, step


def test_init_sets_basics():
    s = init_state(
        4,
        3,
        snake=[(2, 2), (1, 2), (0, 2)],
        food={"apple": (0, 0), "big": (3, 0)},
        direction="R",
        wrap=False,
        obstacles=[(3, 2)],
    )
    assert s["alive"] is True
    assert s["score"] == 0
    assert s["steps"] == 0
    assert s["direction"] == "R"
    assert s["width"] == 4 and s["height"] == 3
    assert set(s["obstacles"]) == {(3, 2)}
    assert set(s["food"].keys()) == {"apple", "big"}


def test_step_moves_one_cell():
    s = init_state(
        5,
        5,
        snake=[(2, 2), (1, 2), (0, 2)],
        food={"apple": (4, 4), "big": (4, 3)},
        direction="R",
    )
    s2 = step(s, "R")
    assert s2["snake"][0] == (3, 2)
    assert s2["steps"] == 1
    assert s2["alive"] is True


def test_reverse_direction_is_ignored():
    s = init_state(
        5,
        5,
        snake=[(2, 2), (1, 2), (0, 2)],
        food={"apple": (4, 4), "big": (4, 3)},
        direction="R",
    )
    s2 = step(s, "L")  # opposite, should keep moving right
    assert s2["snake"][0] == (3, 2)
    assert s2["direction"] == "R"


def test_apple_eat_increases_score_and_triggers_respawn():
    s = init_state(
        3,
        3,
        snake=[(1, 1), (0, 1)],
        food={"apple": (2, 1), "big": (2, 2)},
        direction="R",
    )
    s2 = step(s, "R")
    assert s2["score"] == 1
    # pending growth was applied; length should be +1 now (kept tail)
    assert len(s2["snake"]) == 3
    # apple respawns to first empty cell scanning L->R, T->B
    # occupied: snake + big at (2,2)
    assert s2["food"]["apple"] == (0, 0)


def test_wall_collision_kills_when_wrap_false():
    s = init_state(
        3,
        3,
        snake=[(2, 1), (1, 1), (0, 1)],
        food={"apple": (0, 0), "big": (0, 2)},
        direction="R",
        wrap=False,
    )
    s2 = step(s, "R")
    assert s2["alive"] is False
