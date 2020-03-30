import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.rmsprop import RMSProp


def test_RMSProp_quadratic():
    assert np.allclose(RMSProp(quad1, max_iter=2000).minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(RMSProp(quad2, max_iter=4000).minimize()[0], quad2.x_star(), rtol=0.1)


def test_RMSProp_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(RMSProp(obj, max_iter=1500).minimize()[0], obj.x_star(), rtol=0.1)


def test_RMSProp_standard_momentum_quadratic():
    assert np.allclose(RMSProp(quad1, momentum_type='standard').minimize()[0], quad1.x_star(), rtol=1e-3)
    assert np.allclose(RMSProp(quad2, momentum_type='standard').minimize()[0], quad2.x_star(), rtol=1e-3)


def test_RMSProp_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(RMSProp(obj, momentum_type='standard').minimize()[0], obj.x_star(), rtol=1e-3)


def test_RMSProp_nesterov_momentum_quadratic():
    assert np.allclose(RMSProp(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star(), rtol=1e-3)
    assert np.allclose(RMSProp(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star(), rtol=1e-4)


def test_RMSProp_nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(RMSProp(obj, momentum_type='nesterov').minimize()[0], obj.x_star(), rtol=1e-3)


if __name__ == "__main__":
    pytest.main()
