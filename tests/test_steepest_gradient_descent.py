import numpy as np
import pytest

from functions import *
from steepest_gradient_descent import SDQ, SDG


def test_quadratic_functions_SDQ():
    x0 = [[-1], [1]]

    x, status = SDQ(gen_quad_1, x0)
    assert np.allclose(x, [[-2.1875], [-1.5625]])
    assert status is 'optimal'

    x, status = SDQ(gen_quad_2, x0)
    assert np.allclose(x, [[-4.0625], [-3.4375]])
    assert status is 'optimal'

    # TODO fixed
    x, status = SDQ(gen_quad_3, x0)
    assert np.allclose(x, [[-2062.22589532], [-2060.22589532]])
    assert status is 'stopped'

    # TODO fixed
    # x, status = SDQ(gen_quad_4, x0)
    # assert np.allclose(x, [[np.nan], [np.nan]])
    # assert status is 'stopped'

    x, status = SDQ(gen_quad_5, x0)
    assert np.allclose(x, [[-3.7625], [-3.7375]])
    assert status is 'optimal'


def test_quadratic_functions_SDG():
    x0 = [[-1], [1]]

    x, status = SDG(gen_quad_1, x0)
    assert np.allclose(x, [[-2.1875], [-1.5625]])
    assert status is 'optimal'

    x, status = SDG(gen_quad_2, x0)
    assert np.allclose(x, [[-4.0625], [-3.4375]])
    assert status is 'optimal'

    # TODO fixed
    x, status = SDG(gen_quad_3, x0)
    assert np.allclose(x, [[-1031.61294766], [-1029.61294766]])
    assert status is 'stopped'

    # TODO fixed
    x, status = SDG(gen_quad_4, x0)
    assert np.allclose(x, [[-4.90680224e+153], [-3.63129102e+153]])
    assert status is 'stopped'

    x, status = SDG(gen_quad_5, x0)
    assert np.allclose(x, [[-3.7625], [-3.7375]])
    assert status is 'optimal'


def test_Rosenbrock_SDG():
    x0 = [[-1], [1]]

    # TODO fixed
    # x, status = SDG(Rosenbrock(), x0)
    # assert np.allclose(x, [[1.0080], [1.0162]])
    # assert status is 'optimal'


if __name__ == "__main__":
    pytest.main()
