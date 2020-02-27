import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.adadelta import AdaDelta


def test_AdaDelta_quadratic():
    x, _ = AdaDelta(quad1, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad1.x_star(), rtol=0.01)

    x, _ = AdaDelta(quad2, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad2.x_star())


def test_AdaDelta_Rosenbrock():
    obj = Rosenbrock()
    x, _ = AdaDelta(obj, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.01)


if __name__ == "__main__":
    pytest.main()
