import pytest
from eval.passk import pass_at_k


def test_pass_at_k_basic_values():
    # n=5, c=1
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-12
    # n=5, c=2
    assert abs(pass_at_k(5, 2, 1) - 0.4) < 1e-12
    # n=5, c=2, k=3 => 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
    assert abs(pass_at_k(5, 2, 3) - 0.9) < 1e-12


def test_pass_at_k_edge_cases():
    assert pass_at_k(3, 3, 1) == 1.0
    assert pass_at_k(3, 1, 3) == 1.0  # (n-c)=2 < k=3 => guaranteed
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 2, 0)
