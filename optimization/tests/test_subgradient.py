import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.subgradient import Subgradient


def test_quadratic():
    x, _ = Subgradient(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = Subgradient(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = Subgradient(obj).minimize()
    assert np.allclose(x, obj.x_star())


if __name__ == "__main__":
    pytest.main()
