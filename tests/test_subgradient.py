import numpy as np
import pytest

import utils
from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.subgradient import Subgradient


@utils.not_test
def test_quadratic():
    x, _ = Subgradient(quad1)
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = Subgradient(quad2)
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = Subgradient(quad5)
    assert np.allclose(quad5.jacobian(x), 0)


@utils.not_test
def test_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = Subgradient(obj)
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = Subgradient(obj)
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
