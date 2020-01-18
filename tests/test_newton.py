import numpy as np
import pytest

from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.newton import NWTN


def test_quadratic():
    x, _ = NWTN(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = NWTN(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = NWTN(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = NWTN(obj).minimize()
    assert np.allclose(x, obj.x_star)


if __name__ == "__main__":
    pytest.main()
