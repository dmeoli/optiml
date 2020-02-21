import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.gradient_descent import *


def test_SDQ():
    x, _ = SDQ(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = SDQ(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_SDG_quadratic():
    x, _ = SDG(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = SDG(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_SDG_Rosenbrock():
    obj = Rosenbrock()
    x, _ = SDG(obj).minimize()
    assert np.allclose(x, obj.x_star())


def test_GD_quadratic():
    x, _ = GD(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = GD(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_GD_Rosenbrock():
    obj = Rosenbrock()
    x, _ = GD(obj).minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.1)


def test_GD_standard_momentum_quadratic():
    x, _ = GD(quad1, momentum_type='standard').minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = GD(quad2, momentum_type='standard').minimize()
    assert np.allclose(x, quad2.x_star())


def test_GD_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = GD(obj, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star())


def test_GD_Nesterov_momentum_quadratic():
    x, _ = GD(quad1, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = GD(quad2, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad2.x_star())


def test_GD_Nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = GD(obj, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star())


if __name__ == "__main__":
    pytest.main()
