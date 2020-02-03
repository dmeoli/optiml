import numpy as np
import pytest

import utils
from optimization.optimization_function import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.subgradient import SGM


@utils.not_test
def test_quadratic():
    x, _ = SGM(quad1).minimize()
    assert np.allclose(x, quad1.f_star())

    x, _ = SGM(quad2).minimize()
    assert np.allclose(x, quad2.f_star())

    x, _ = SGM(quad5).minimize()
    assert np.allclose(x, quad5.f_star())


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = SGM(obj).minimize()
    assert np.allclose(x, obj.f_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
