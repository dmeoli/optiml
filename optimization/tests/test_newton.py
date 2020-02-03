import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.newton import NWTN


def test_quadratic():
    x, _ = NWTN(quad1).minimize()
    assert np.allclose(x, quad1.f_star())

    x, _ = NWTN(quad2).minimize()
    assert np.allclose(x, quad2.f_star())

    x, _ = NWTN(quad5).minimize()
    assert np.allclose(x, quad5.f_star())


def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = NWTN(obj).minimize()
    assert np.allclose(x, obj.f_star())


if __name__ == "__main__":
    pytest.main()
