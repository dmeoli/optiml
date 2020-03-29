import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.proximal_bundle import ProximalBundle


def test_quadratic():
    assert np.allclose(ProximalBundle(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(ProximalBundle(quad2).minimize()[0], quad2.x_star())


def test_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(ProximalBundle(obj).minimize()[0], obj.x_star())


def test_standard_momentum_quadratic():
    assert np.allclose(ProximalBundle(quad1, momentum_type='standard').minimize()[0], quad1.x_star())
    assert np.allclose(ProximalBundle(quad2).minimize()[0], quad2.x_star())


def test_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(ProximalBundle(obj, momentum_type='standard').minimize()[0], obj.x_star())


def test_Nesterov_momentum_quadratic():
    assert np.allclose(ProximalBundle(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star())
    assert np.allclose(ProximalBundle(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star())


def test_Nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(ProximalBundle(obj, momentum_type='nesterov').minimize()[0], obj.x_star())


if __name__ == "__main__":
    pytest.main()
