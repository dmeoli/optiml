import numpy as np
import pytest

from optimization.test_functions import gen_quad_1, gen_quad_2, gen_quad_5, Rosenbrock
from optimization.unconstrained.gradient_descent import SteepestGradientDescentQuadratic, SteepestGradientDescent, \
    GradientDescent


def test_SteepestGradientDescentQuadratic_quadratic():
    x, status = SteepestGradientDescentQuadratic(gen_quad_1).minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = SteepestGradientDescentQuadratic(gen_quad_2).minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    # x, status = SteepestGradientDescentQuadratic(gen_quad_3).minimize()
    # assert np.allclose(x, gen_quad_3.x_star)
    # assert status is 'optimal'
    #
    # x, status = SteepestGradientDescentQuadratic(gen_quad_4).minimize()
    # assert np.allclose(x, gen_quad_4.x_star)
    # assert status is 'optimal'

    x, status = SteepestGradientDescentQuadratic(gen_quad_5).minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_SteepestGradientDescent_quadratic():
    x, status = SteepestGradientDescent(gen_quad_1).minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = SteepestGradientDescent(gen_quad_2).minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    # x, status = SteepestGradientDescent(gen_quad_3).minimize()
    # assert np.allclose(x, gen_quad_3.x_star)
    # assert status is 'optimal'
    #
    # x, status = SteepestGradientDescent(gen_quad_4).minimize()
    # assert np.allclose(x, gen_quad_4.x_star)
    # assert status is 'optimal'

    x, status = SteepestGradientDescent(gen_quad_5).minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_SteepestGradientDescent_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = SteepestGradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'stopped'

    obj = Rosenbrock(autodiff=False)
    x, status = SteepestGradientDescent(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'stopped'


def test_GradientDescent_quadratic():
    x, status = GradientDescent(gen_quad_1, step_rate=0.01).minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_2, step_rate=0.01).minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    # x, status = GradientDescent(gen_quad_3).minimize()
    # assert np.allclose(x, gen_quad_3.x_star)
    # assert status is 'optimal'
    #
    # x, status = GradientDescent(gen_quad_4).minimize()
    # assert np.allclose(x, gen_quad_4.x_star)
    # assert status is 'optimal'

    x, status = GradientDescent(gen_quad_5, step_rate=0.01).minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_GradientDescent_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = GradientDescent(obj, step_rate=0.01).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = GradientDescent(obj, step_rate=0.01).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'


def test_GradientDescent_standard_quadratic():
    x, status = GradientDescent(gen_quad_1, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_2, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    # x, status = GradientDescent(gen_quad_3, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    # assert np.allclose(x, gen_quad_3.x_star)
    # assert status is 'optimal'
    #
    # x, status = GradientDescent(gen_quad_4, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    # assert np.allclose(x, gen_quad_4.x_star)
    # assert status is 'optimal'

    x, status = GradientDescent(gen_quad_5, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_GradientDescent_standard_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'


def test_GradientDescent_Nesterov_quadratic():
    x, status = GradientDescent(gen_quad_1, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_2, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    # x, status = GradientDescent(gen_quad_3, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    # assert np.allclose(x, gen_quad_3.x_star)
    # assert status is 'optimal'
    #
    # x, status = GradientDescent(gen_quad_4, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    # assert np.allclose(x, gen_quad_4.x_star)
    # assert status is 'optimal'

    x, status = GradientDescent(gen_quad_5, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_GradientDescent_Nesterov_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'


if __name__ == "__main__":
    pytest.main()
