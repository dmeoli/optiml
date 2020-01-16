import numpy as np
import pytest

from optimization.test_functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.conjugate_gradient import NCG, CGQ


def test_CGQ_quadratic():
    x, _ = CGQ(quad1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = CGQ(quad2).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = CGQ(quad5).minimize()
    assert np.allclose(x, quad5.x_star)


def test_NCG_quadratic():
    x, _ = NCG(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = NCG(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = NCG(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_NCG_Rosenbrock():
    obj = Rosenbrock(autodiff=True)
    x, _ = NCG(obj, wf=4).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)

    obj = Rosenbrock(autodiff=False)
    x, _ = NCG(obj, wf=4).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
