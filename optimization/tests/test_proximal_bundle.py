import numpy as np
import pytest

import utils
from optimization.functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.proximal_bundle import PBM


@utils.not_test
def test_quadratic():
    x, _ = PBM(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = PBM(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = PBM(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = PBM(obj).minimize()
    assert np.allclose(x, obj.x_star)


if __name__ == "__main__":
    pytest.main()
