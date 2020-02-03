import numpy as np
import pytest

import utils
from optimization.optimization_function import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.proximal_bundle import PBM


@utils.not_test
def test_quadratic():
    x, _ = PBM(quad1).minimize()
    assert np.allclose(x, quad1.f_star())

    x, _ = PBM(quad2).minimize()
    assert np.allclose(x, quad2.f_star())

    x, _ = PBM(quad5).minimize()
    assert np.allclose(x, quad5.f_star())


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = PBM(obj).minimize()
    assert np.allclose(x, obj.f_star())


if __name__ == "__main__":
    pytest.main()
