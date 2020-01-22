import numpy as np
import pytest

from optimization.functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.adam import Adam


def test_Adam_quadratic():
    x, _ = Adam(quad1).minimize()
    np.allclose(quad1.jacobian(x), 0)

    x, _ = Adam(quad2).minimize()
    np.allclose(quad2.jacobian(x), 0)

    x, _ = Adam(quad5).minimize()
    np.allclose(quad5.jacobian(x), 0)


def test_Adam_Rosenbrock():
    obj = Rosenbrock()
    x, _ = Adam(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
