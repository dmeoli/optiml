import pytest

import utils
from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.gradient_descent import *


def test_SteepestGradientDescentQuadratic():
    x, _ = SteepestGradientDescentQuadratic(quad1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = SteepestGradientDescentQuadratic(quad2).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = SteepestGradientDescentQuadratic(quad5).minimize()
    assert np.allclose(x, quad5.x_star)


def test_SteepestGradientDescent_quadratic():
    x, _ = SteepestGradientDescent(quad1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = SteepestGradientDescent(quad2).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = SteepestGradientDescent(quad5).minimize()
    assert np.allclose(x, quad5.x_star)


def test_SteepestGradientDescent_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = SteepestGradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = SteepestGradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


def test_GradientDescent_quadratic():
    x, _ = GradientDescent(quad1, step_rate=0.01).minimize()
    np.allclose(quad1.jacobian(x), 0)

    x, _ = GradientDescent(quad2, step_rate=0.01).minimize()
    np.allclose(quad2.jacobian(x), 0)

    x, _ = GradientDescent(quad5, step_rate=0.01).minimize()
    np.allclose(quad5.jacobian(x), 0)


@utils.not_test
# TODO try with schedule
def test_GradientDescent_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = GradientDescent(obj, step_rate=0.01).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = GradientDescent(obj, step_rate=0.01).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


def test_GradientDescent_standard_quadratic():
    x, _ = GradientDescent(quad1, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    np.allclose(quad1.jacobian(x), 0)

    x, _ = GradientDescent(quad2, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    np.allclose(quad2.jacobian(x), 0)

    x, _ = GradientDescent(quad5, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    np.allclose(quad5.jacobian(x), 0)


@utils.not_test
# TODO try with schedule
def test_GradientDescent_standard_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


def test_GradientDescent_Nesterov_quadratic():
    x, _ = GradientDescent(quad1, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    np.allclose(quad1.jacobian(x), 0)

    x, _ = GradientDescent(quad2, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    np.allclose(quad2.jacobian(x), 0)

    x, _ = GradientDescent(quad5, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    np.allclose(quad5.jacobian(x), 0)


@utils.not_test
# TODO try with schedule
def test_GradientDescent_Nesterov_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
