import pytest

from calculator import evaluate


def test_basic_addition():
    assert evaluate("2 + 3") == pytest.approx(5.0)


def test_operator_precedence():
    assert evaluate("2 + 3 * 4") == pytest.approx(14.0)


def test_parentheses():
    assert evaluate("(2 + 3) * 4") == pytest.approx(20.0)


def test_simple_variable():
    assert evaluate("x + 1", {"x": 5}) == pytest.approx(6.0)


def test_division_by_zero():
    with pytest.raises(ValueError, match="division by zero"):
        evaluate("1 / 0")


def test_unary_minus():
    assert evaluate("-5") == pytest.approx(-5.0)
