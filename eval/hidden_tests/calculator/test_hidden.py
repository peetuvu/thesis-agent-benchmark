import pytest

from calculator import evaluate


def test_right_associative_exponent():
    assert evaluate("2 ** 3 ** 2") == pytest.approx(512.0)


def test_unary_minus_vs_exponent_precedence():
    assert evaluate("-3 ** 2") == pytest.approx(-9.0)


def test_parenthesized_negative_exponent():
    assert evaluate("(-3) ** 2") == pytest.approx(9.0)


def test_nested_parentheses():
    assert evaluate("((1 + 2) * (3 + 4))") == pytest.approx(21.0)


def test_chained_subtraction_left_assoc():
    assert evaluate("10 - 3 - 2") == pytest.approx(5.0)


def test_chained_division_left_assoc():
    assert evaluate("24 / 4 / 2") == pytest.approx(3.0)


def test_modulo():
    assert evaluate("7 % 3") == pytest.approx(1.0)


def test_modulo_by_zero():
    with pytest.raises(ValueError, match="division by zero"):
        evaluate("5 % 0")


def test_float_numbers():
    assert evaluate(".5 + 3.14") == pytest.approx(3.64)


def test_undefined_variable():
    with pytest.raises(ValueError, match="x"):
        evaluate("x + 1")


def test_empty_expression():
    with pytest.raises(ValueError):
        evaluate("")


def test_whitespace_only():
    with pytest.raises(ValueError):
        evaluate("   ")


def test_complex_nested_negation():
    assert evaluate("-(2 + 3) ** 2") == pytest.approx(-25.0)


def test_variable_in_complex_expr():
    assert evaluate("x * (y + 1)", {"x": 3, "y": 4}) == pytest.approx(15.0)


def test_division_by_zero_in_subexpr():
    with pytest.raises(ValueError, match="division by zero"):
        evaluate("1 + 2 / (3 - 3)")


def test_invalid_character():
    with pytest.raises(ValueError):
        evaluate("2 & 3")


def test_unary_minus_with_addition():
    assert evaluate("2 + -3") == pytest.approx(-1.0)


def test_power_of_negative_exponent():
    assert evaluate("2 ** -3") == pytest.approx(0.125)


def test_whitespace_variations():
    assert evaluate("  2  +  3  ") == pytest.approx(5.0)
