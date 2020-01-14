import numpy as np
import pytest

from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.newton import Newton


def test_quadratic():
    x, _ = Newton(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = Newton(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = Newton(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = Newton(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = Newton(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
