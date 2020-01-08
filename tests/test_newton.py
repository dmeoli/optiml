import numpy as np
import pytest

from optimization.test_functions import *
from optimization.unconstrained.newton import Newton


def test_quadratic():
    x, status = Newton(gen_quad_1).minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = Newton(gen_quad_2).minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    # x, status = Newton(gen_quad_3).minimize()
    # assert np.allclose(x, gen_quad_3.x_star)
    # assert status is 'optimal'

    # x, status = Newton(gen_quad_4).minimize()
    # assert np.allclose(x, gen_quad_4.x_star)
    # assert status is 'optimal'

    x, status = Newton(gen_quad_5).minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = Newton(obj).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = Newton(obj).minimize()
    assert np.allclose(x, obj.x_star)
    assert status is 'optimal'


if __name__ == "__main__":
    pytest.main()
