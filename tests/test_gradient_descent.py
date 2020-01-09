import numpy as np
import pytest

import utils
from optimization.test_functions import gen_quad_1, gen_quad_2, gen_quad_5, Rosenbrock
from optimization.unconstrained.gradient_descent import SteepestGradientDescentQuadratic, SteepestGradientDescent, \
    GradientDescent


def test_SteepestGradientDescentQuadratic_quadratic():
    x, _ = SteepestGradientDescentQuadratic(gen_quad_1).minimize()
    assert np.allclose(gen_quad_1.jacobian(x), 0)

    x, _ = SteepestGradientDescentQuadratic(gen_quad_2).minimize()
    np.allclose(gen_quad_2.jacobian(x), 0)

    # x, _ = SteepestGradientDescentQuadratic(gen_quad_3).minimize()
    # np.allclose(gen_quad_3.jacobian(x), 0)

    # x, _ = SteepestGradientDescentQuadratic(gen_quad_4).minimize()
    # np.allclose(gen_quad_4.jacobian(x), 0)

    x, _ = SteepestGradientDescentQuadratic(gen_quad_5).minimize()
    np.allclose(gen_quad_5.jacobian(x), 0)


def test_SteepestGradientDescent_quadratic():
    x, _ = SteepestGradientDescent(gen_quad_1).minimize()
    np.allclose(gen_quad_1.jacobian(x), 0)

    x, _ = SteepestGradientDescent(gen_quad_2).minimize()
    np.allclose(gen_quad_2.jacobian(x), 0)

    # x, _ = SteepestGradientDescent(gen_quad_3).minimize()
    # np.allclose(gen_quad_3.jacobian(x), 0)

    # x, _ = SteepestGradientDescent(gen_quad_4).minimize()
    # np.allclose(gen_quad_4.jacobian(x), 0)

    x, _ = SteepestGradientDescent(gen_quad_5).minimize()
    np.allclose(gen_quad_5.jacobian(x), 0)


@utils.not_test
def test_SteepestGradientDescent_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = SteepestGradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = SteepestGradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


@utils.not_test
def test_GradientDescent_quadratic():
    x, _ = GradientDescent(gen_quad_1, step_rate=0.01).minimize()
    np.allclose(gen_quad_1.jacobian(x), 0)

    x, _ = GradientDescent(gen_quad_2, step_rate=0.01).minimize()
    np.allclose(gen_quad_2.jacobian(x), 0)

    # x, _ = GradientDescent(gen_quad_3).minimize()
    # np.allclose(gen_quad_3.jacobian(x), 0)

    # x, _ = GradientDescent(gen_quad_4).minimize()
    # np.allclose(gen_quad_4.jacobian(x), 0)

    x, _ = GradientDescent(gen_quad_5, step_rate=0.01).minimize()
    np.allclose(gen_quad_5.jacobian(x), 0)


@utils.not_test
def test_GradientDescent_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = GradientDescent(obj, step_rate=0.01).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = GradientDescent(obj, step_rate=0.01).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


@utils.not_test
def test_GradientDescent_standard_quadratic():
    x, _ = GradientDescent(gen_quad_1, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    np.allclose(gen_quad_1.jacobian(x), 0)

    x, _ = GradientDescent(gen_quad_2, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    np.allclose(gen_quad_2.jacobian(x), 0)

    # x, _ = GradientDescent(gen_quad_3, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    # np.allclose(gen_quad_3.jacobian(x), 0)

    # x, _ = GradientDescent(gen_quad_4, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    # np.allclose(gen_quad_4.jacobian(x), 0)

    x, _ = GradientDescent(gen_quad_5, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    np.allclose(gen_quad_5.jacobian(x), 0)


@utils.not_test
def test_GradientDescent_standard_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


@utils.not_test
def test_GradientDescent_Nesterov_quadratic():
    x, _ = GradientDescent(gen_quad_1, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    np.allclose(gen_quad_1.jacobian(x), 0)

    x, _ = GradientDescent(gen_quad_2, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    np.allclose(gen_quad_2.jacobian(x), 0)

    # x, _ = GradientDescent(gen_quad_3, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    # np.allclose(gen_quad_3.jacobian(x), 0)

    # x, _ = GradientDescent(gen_quad_4, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    # np.allclose(gen_quad_4.jacobian(x), 0)

    x, _ = GradientDescent(gen_quad_5, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    np.allclose(gen_quad_5.jacobian(x), 0)


@utils.not_test
def test_GradientDescent_Nesterov_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
