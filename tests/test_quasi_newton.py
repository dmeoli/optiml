import numpy as np
import pytest

from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.quasi_newton import BroydenFletcherGoldfarbShanno


def test_quadratic():
    x, _ = BroydenFletcherGoldfarbShanno(quad1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = BroydenFletcherGoldfarbShanno(quad2).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = BroydenFletcherGoldfarbShanno(quad5).minimize()
    assert np.allclose(x, quad5.x_star)


def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = BroydenFletcherGoldfarbShanno(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = BroydenFletcherGoldfarbShanno(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
