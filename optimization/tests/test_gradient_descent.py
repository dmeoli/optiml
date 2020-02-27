import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.gradient_descent import *


def test_SteepestGradientDescentQuadratic():
    x, _ = SteepestGradientDescentQuadratic(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = SteepestGradientDescentQuadratic(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_SteepestGradientDescent_quadratic():
    x, _ = SteepestGradientDescent(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = SteepestGradientDescent(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_SteepestGradientDescent_Rosenbrock():
    obj = Rosenbrock()
    x, _ = SteepestGradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star())


def test_GradientDescent_quadratic():
    x, _ = GradientDescent(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = GradientDescent(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


def test_GradientDescent_Rosenbrock():
    obj = Rosenbrock()
    x, _ = GradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.1)


def test_GradientDescent_standard_momentum_quadratic():
    x, _ = GradientDescent(quad1, momentum_type='standard').minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = GradientDescent(quad2, momentum_type='standard').minimize()
    assert np.allclose(x, quad2.x_star())


def test_GradientDescent_standard_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = GradientDescent(obj, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star())


def test_GradientDescent_Nesterov_momentum_quadratic():
    x, _ = GradientDescent(quad1, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = GradientDescent(quad2, momentum_type='nesterov').minimize()
    assert np.allclose(x, quad2.x_star())


def test_GradientDescent_Nesterov_momentum_Rosenbrock():
    obj = Rosenbrock()
    x, _ = GradientDescent(obj, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star())


if __name__ == "__main__":
    pytest.main()
