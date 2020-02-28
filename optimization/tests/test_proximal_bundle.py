import numpy as np
import pytest

import utils
from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.proximal_bundle import ProximalBundle


@utils.not_test
def test_quadratic():
    assert np.allclose(ProximalBundle(quad1).minimize()[0], quad1.x_star())
    assert np.allclose(ProximalBundle(quad2).minimize()[0], quad2.x_star())


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock()
    assert np.allclose(ProximalBundle(obj).minimize()[0], obj.x_star())


if __name__ == "__main__":
    pytest.main()
