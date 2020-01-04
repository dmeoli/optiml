import pytest

from optimization_test_functions import *
from unconstrained.gradient_descent import *


def test_quadratic_functions_SteepestGradientDescentQuadratic():
    x0 = [[-1], [1]]

    x, status = SteepestGradientDescentQuadratic(gen_quad_1, x0).minimize()
    assert np.allclose(x, [[2.1875], [1.5625]])
    assert status is 'optimal'

    x, status = SteepestGradientDescentQuadratic(gen_quad_2, x0).minimize()
    assert np.allclose(x, [[4.0625], [3.4375]])
    assert status is 'optimal'

    x, status = SteepestGradientDescentQuadratic(gen_quad_5, x0).minimize()
    assert np.allclose(x, [[3.7625], [3.7375]])
    assert status is 'optimal'


def test_quadratic_functions_SteepestGradientDescent():
    x0 = [[-1], [1]]

    x, status = SteepestGradientDescent(gen_quad_1, x0).minimize()
    assert np.allclose(x, [[2.1875], [1.5625]])
    assert status is 'optimal'

    x, status = SteepestGradientDescent(gen_quad_2, x0).minimize()
    assert np.allclose(x, [[4.0625], [3.4375]])
    assert status is 'optimal'

    x, status = SteepestGradientDescent(gen_quad_5, x0).minimize()
    assert np.allclose(x, [[3.7625], [3.7375]])
    assert status is 'optimal'


def test_quadratic_functions_GradientDescent():
    x0 = [[-1], [1]]

    x, status = GradientDescent(gen_quad_1, x0, step_rate=0.01)
    assert np.allclose(x, [[2.1875], [1.5625]])
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_2, x0, step_rate=0.01)
    assert np.allclose(x, [[4.0625], [3.4375]])
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_5, x0, step_rate=0.01)
    assert np.allclose(x, [[3.7625], [3.7375]])
    assert status is 'optimal'


def test_quadratic_functions_GradientDescent_standard_momentum():
    x0 = [[-1], [1]]

    x, status = GradientDescent(gen_quad_1, x0, step_rate=0.01, momentum=0.9, momentum_type='standard')
    assert np.allclose(x, [[2.1875], [1.5625]])
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_2, x0, step_rate=0.01, momentum=0.9, momentum_type='standard')
    assert np.allclose(x, [[4.0625], [3.4375]])
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_5, x0, step_rate=0.01, momentum=0.9, momentum_type='standard')
    assert np.allclose(x, [[3.7625], [3.7375]])
    assert status is 'optimal'


def test_quadratic_functions_GradientDescent_Nesterov_momentum():
    x0 = [[-1], [1]]

    x, status = GradientDescent(gen_quad_1, x0, step_rate=0.01, momentum=0.9, momentum_type='nesterov')
    assert np.allclose(x, [[2.1875], [1.5625]])
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_2, x0, step_rate=0.01, momentum=0.9, momentum_type='nesterov')
    assert np.allclose(x, [[4.0625], [3.4375]])
    assert status is 'optimal'

    x, status = GradientDescent(gen_quad_5, x0, step_rate=0.01, momentum=0.9, momentum_type='nesterov')
    assert np.allclose(x, [[3.7625], [3.7375]])
    assert status is 'optimal'


def test_Rosenbrock_SteepestGradientDescent():
    x0 = [[-1], [1]]

    # TODO fixed: in MATLAB it works with just 1000 max_f_eval
    x, status = SteepestGradientDescent(Rosenbrock(), x0, max_f_eval=10000).minimize()
    assert np.allclose(x, [[1], [1]])
    assert status is 'optimal'


def test_Rosenbrock_GradientDescent():
    x0 = [[-1], [1]]

    x, status = GradientDescent(Rosenbrock(), x0, step_rate=0.01)
    assert np.allclose(x, [[1], [1]])
    assert status is 'optimal'


def test_Rosenbrock_GradientDescent_standard_momentum():
    x0 = [[-1], [1]]

    x, status = GradientDescent(Rosenbrock(), x0, step_rate=0.01, momentum=0.9, momentum_type='standard')
    assert np.allclose(x, [[1], [1]])
    assert status is 'optimal'


def test_Rosenbrock_GradientDescent_Nesterov_momentum():
    x0 = [[-1], [1]]

    x, status = GradientDescent(Rosenbrock(), x0, step_rate=0.01, momentum=0.9, momentum_type='nesterov')
    assert np.allclose(x, [[1], [1]])
    assert status is 'optimal'


if __name__ == "__main__":
    pytest.main()
