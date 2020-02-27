import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.rmsprop import RMSProp


def test_RMSProp_quadratic():
    x, _ = RMSProp(quad1, max_iter=2000).minimize()
    assert np.allclose(x, quad1.x_star(), rtol=0.1)

    x, _ = RMSProp(quad2, max_iter=4000).minimize()
    assert np.allclose(x, quad2.x_star(), rtol=0.1)


def test_RMSProp_Rosenbrock():
    obj = Rosenbrock()
    x, _ = RMSProp(obj).minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.1)


def test_RMSProp_standard_momentum_quadratic():
    x, _ = RMSProp(quad1, momentum_type='standard').minimize()
    assert np.allclose(x, quad1.x_star(), rtol=1e-3)

    x, _ = RMSProp(quad2, momentum_type='standard').minimize()
    assert np.allclose(x, quad2.x_star(), rtol=1e-3)


def test_RMSProp_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = RMSProp(obj, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star(), rtol=1e-3)


def test_RMSProp_nesterov_momentum_quadratic():
    x, _ = RMSProp(quad1, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad1.x_star(), rtol=1e-3)

    x, _ = RMSProp(quad2, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad2.x_star(), rtol=1e-4)


def test_RMSProp_nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = RMSProp(obj, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star(), rtol=1e-3)


if __name__ == "__main__":
    pytest.main()
