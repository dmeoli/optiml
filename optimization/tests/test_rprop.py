import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.rprop import RProp


def test_RProp_quadratic():
    x, _ = RProp(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = RProp(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_RProp_Rosenbrock():
    obj = Rosenbrock()
    x, _ = RProp(obj).minimize()
    assert np.allclose(x, obj.x_star(), rtol=1e-4)


if __name__ == "__main__":
    pytest.main()
