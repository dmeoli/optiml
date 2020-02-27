import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.adamax import AdaMax


def test_AdaMax_quadratic():
    x, _ = AdaMax(quad1, nesterov_momentum=True).minimize()
    assert np.allclose(x, quad1.x_star(), rtol=0.1)

    x, _ = AdaMax(quad2, nesterov_momentum=True).minimize()
    assert np.allclose(x, quad2.x_star(), rtol=0.1)


def test_AdaMax_Rosenbrock():
    obj = Rosenbrock()
    x, _ = AdaMax(obj, nesterov_momentum=True).minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.2)


if __name__ == "__main__":
    pytest.main()
