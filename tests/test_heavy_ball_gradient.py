import numpy as np
import pytest

from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.heavy_ball_gradient import HBG


def test_quadratic():
    x, _ = HBG(quad1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = HBG(quad2).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = HBG(quad5).minimize()
    assert np.allclose(x, quad5.x_star, rtol=1e-4)


def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = HBG(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
