import numpy as np
import pytest

import utils
from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.proximal_bundle import ProximalBundle


@utils.not_test
def test_quadratic():
    x, _ = ProximalBundle(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = ProximalBundle(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = ProximalBundle(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = ProximalBundle(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = ProximalBundle(obj).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()