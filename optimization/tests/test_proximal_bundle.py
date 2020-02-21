import numpy as np
import pytest

import utils
from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.proximal_bundle import PBM


@utils.not_test
def test_quadratic():
    x, _ = PBM(quad1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = PBM(quad2).minimize()
    assert np.allclose(x, quad2.x_star())


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = PBM(obj).minimize()
    assert np.allclose(x, obj.x_star())


if __name__ == "__main__":
    pytest.main()
