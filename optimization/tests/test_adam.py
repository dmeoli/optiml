import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.adam import Adam


def test_Adam_quadratic():
    x, _ = Adam(quad1).minimize()
    np.allclose(x, quad1.x_star())

    x, _ = Adam(quad2).minimize()
    np.allclose(x, quad2.x_star())

    x, _ = Adam(quad5).minimize()
    np.allclose(x, quad5.x_star())


def test_Adam_Rosenbrock():
    obj = Rosenbrock()
    x, _ = Adam(obj).minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
