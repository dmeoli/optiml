import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.adam import Adam


def test_Nadam_quadratic():
    x, _ = Adam(quad1, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad1.x_star(), rtol=0.1)

    x, _ = Adam(quad2, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad2.x_star(), rtol=0.01)


def test_Nadam_Rosenbrock():
    obj = Rosenbrock()
    x, _ = Adam(obj, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.15)


if __name__ == "__main__":
    pytest.main()
