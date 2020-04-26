import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.stochastic.rprop import RProp


def test_RProp_quadratic():
    assert np.allclose(RProp(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(RProp(quad2).minimize()[0], quad2.x_star())


def test_RProp_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(RProp(obj).minimize()[0], obj.x_star(), rtol=0.1)


def test_RProp_standard_momentum_quadratic():
    assert np.allclose(RProp(quad1, momentum_type='standard', momentum=0.6).minimize()[0], quad1.x_star())
    assert np.allclose(RProp(quad2, momentum_type='standard', momentum=0.6).minimize()[0], quad2.x_star())


def test_RProp_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(RProp(obj, momentum_type='standard', momentum=0.6).minimize()[0], obj.x_star(), rtol=0.1)


def test_RProp_nesterov_momentum_quadratic():
    assert np.allclose(RProp(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star())
    assert np.allclose(RProp(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star())


def test_RProp_nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(RProp(obj, momentum_type='nesterov').minimize()[0], obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
