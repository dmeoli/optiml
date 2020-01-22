import numpy as np
import pytest

import utils
from optimization.functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.subgradient import SGM


@utils.not_test
def test_quadratic():
    x, _ = SGM(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = SGM(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = SGM(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = SGM(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
