import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.gradient_descent import *


def test_SteepestGradientDescentQuadratic():
    assert np.allclose(SteepestGradientDescentQuadratic(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(SteepestGradientDescentQuadratic(quad2).minimize()[0], quad2.x_star())


def test_SteepestGradientDescent_quadratic():
    assert np.allclose(SteepestGradientDescent(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(SteepestGradientDescent(quad2).minimize()[0], quad2.x_star())


def test_SteepestGradientDescent_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(SteepestGradientDescent(obj).minimize()[0], obj.x_star())


def test_GradientDescent_quadratic():
    assert np.allclose(GradientDescent(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(GradientDescent(quad2).minimize()[0], quad2.x_star())


def test_GradientDescent_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(GradientDescent(obj).minimize()[0], obj.x_star(), rtol=0.1)


def test_GradientDescent_standard_momentum_quadratic():
    assert np.allclose(GradientDescent(quad1, momentum_type='standard').minimize()[0], quad1.x_star())
    assert np.allclose(GradientDescent(quad2, momentum_type='standard').minimize()[0], quad2.x_star())


def test_GradientDescent_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(GradientDescent(obj, momentum_type='standard').minimize()[0], obj.x_star())


def test_GradientDescent_Nesterov_momentum_quadratic():
    assert np.allclose(GradientDescent(quad1, momentum_type='nesterov').minimize()[0], quad1.x_star())
    assert np.allclose(GradientDescent(quad2, momentum_type='nesterov').minimize()[0], quad2.x_star())


def test_GradientDescent_Nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(GradientDescent(obj, momentum_type='nesterov').minimize()[0], obj.x_star())


if __name__ == "__main__":
    pytest.main()
