import numpy as np
import pytest

from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.quasi_newton import BroydenFletcherGoldfarbShanno


def test_quadratic():
    x, _ = BroydenFletcherGoldfarbShanno(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = BroydenFletcherGoldfarbShanno(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = BroydenFletcherGoldfarbShanno(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = BroydenFletcherGoldfarbShanno(obj).minimize()
    assert np.allclose(x, obj.x_star)

    obj = Rosenbrock(autodiff=False)
    x, _ = BroydenFletcherGoldfarbShanno(obj).minimize()
    assert np.allclose(x, obj.x_star)


if __name__ == "__main__":
    pytest.main()
