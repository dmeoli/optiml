import numpy as np
import pytest

from optimization import gen_quad_1, gen_quad_2, gen_quad_5, Rosenbrock
from optimization.unconstrained.heavy_ball_gradient import HeavyBallGradient


def test_quadratic():
    x, status = HeavyBallGradient(gen_quad_1).minimize()
    assert np.allclose(x, gen_quad_1.x_star)
    assert status is 'optimal'

    x, status = HeavyBallGradient(gen_quad_2).minimize()
    assert np.allclose(x, gen_quad_2.x_star)
    assert status is 'optimal'

    # x, status = HeavyBallGradient(gen_quad_3).minimize()
    # assert np.allclose(x, gen_quad_3.x_star)
    # assert status is 'optimal'

    # x, status = HeavyBallGradient(gen_quad_4).minimize()
    # assert np.allclose(x, gen_quad_4.x_star)
    # assert status is 'optimal'

    x, status = HeavyBallGradient(gen_quad_5).minimize()
    assert np.allclose(x, gen_quad_5.x_star)
    assert status is 'optimal'


def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, status = HeavyBallGradient(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'

    obj = Rosenbrock(autodiff=False)
    x, status = HeavyBallGradient(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)
    assert status is 'optimal'


if __name__ == "__main__":
    pytest.main()
