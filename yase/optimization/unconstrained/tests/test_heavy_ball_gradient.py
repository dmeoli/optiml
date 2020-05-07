import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.line_search.heavy_ball_gradient import HeavyBallGradient


def test_quadratic():
    assert np.allclose(HeavyBallGradient(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(HeavyBallGradient(quad2).minimize()[0], quad2.x_star())


def test_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(HeavyBallGradient(obj).minimize()[0], obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
