import pytest

from optimization.test_functions import *
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
    # TODO fixed: in MATLAB it works with just 1000 max_f_eval
    x, status = SteepestGradientDescent(obj, max_f_eval=10000).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    # TODO fixed: in MATLAB it works with just 1000 max_f_eval
    x, status = SteepestGradientDescent(obj, max_f_eval=10000).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'


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
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = GradientDescent(obj, step_rate=0.01).minimize()
    assert np.allclose(x, obj.x_star)
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
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='standard').minimize()
    assert np.allclose(x, obj.x_star)
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
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = GradientDescent(obj, step_rate=0.01, momentum=0.9, momentum_type='nesterov').minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'


if __name__ == "__main__":
    pytest.main()
