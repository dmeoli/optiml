import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.proximal_bundle import ProximalBundle


def test_quadratic():
    assert np.allclose(ProximalBundle(quad1, solver='CVXOPT').minimize()[0], quad1.x_star())
    assert np.allclose(ProximalBundle(quad2, solver='CVXOPT').minimize()[0], quad2.x_star(), rtol=0.1)


def test_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(ProximalBundle(obj, solver='CVXOPT').minimize()[0], obj.x_star(), rtol=0.1)


def test_standard_momentum_quadratic():
    assert np.allclose(ProximalBundle(quad1, momentum_type='standard', solver='CVXOPT').minimize()[0],
                       quad1.x_star(), rtol=0.1)
    assert np.allclose(ProximalBundle(quad2, momentum_type='standard', solver='CVXOPT').minimize()[0],
                       quad2.x_star(), rtol=0.1)


def test_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(ProximalBundle(obj, momentum_type='standard', momentum=0.8,
                                      max_iter=2000, solver='CVXOPT').minimize()[0], obj.x_star(), rtol=0.1)


def test_Nesterov_momentum_quadratic():
    assert np.allclose(ProximalBundle(quad1, momentum_type='nesterov', momentum=0.2,
                                      solver='CVXOPT').minimize()[0], quad1.x_star(), rtol=0.1)
    assert np.allclose(ProximalBundle(quad2, momentum_type='nesterov', momentum=0.2,
                                      solver='CVXOPT').minimize()[0], quad2.x_star(), rtol=0.1)


def test_Nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(ProximalBundle(obj, momentum_type='nesterov', momentum=0.4,
                                      solver='CVXOPT').minimize()[0], obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
