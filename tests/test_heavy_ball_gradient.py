import numpy as np
import pytest

import utils
from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.heavy_ball_gradient import HeavyBallGradient


def test_quadratic():
    x, _ = HeavyBallGradient(quad1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = HeavyBallGradient(quad2).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = HeavyBallGradient(quad5).minimize()
    assert np.allclose(x, quad5.x_star)


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = HeavyBallGradient(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = HeavyBallGradient(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
