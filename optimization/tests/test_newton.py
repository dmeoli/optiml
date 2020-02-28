import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.newton import Newton


def test_quadratic():
    assert np.allclose(Newton(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(Newton(quad2).minimize()[0], quad2.x_star())


def test_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(Newton(obj).minimize()[0], obj.x_star())


if __name__ == "__main__":
    pytest.main()
