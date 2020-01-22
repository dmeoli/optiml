import numpy as np
import pytest

from optimization.functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.conjugate_gradient import NCG, CGQ


def test_CGQ_quadratic():
    x, _ = CGQ(quad1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = CGQ(quad2).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = CGQ(quad5).minimize()
    assert np.allclose(x, quad5.x_star)


def test_NCG_quadratic_wf0():
    x, _ = NCG(quad1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = NCG(quad2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = NCG(quad5).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_NCG_quadratic_wf1():
    x, _ = NCG(quad1, wf=1).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = NCG(quad2, wf=1).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = NCG(quad5, wf=1).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_NCG_quadratic_wf2():
    x, _ = NCG(quad1, wf=2).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = NCG(quad2, wf=2).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = NCG(quad5, wf=2).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_NCG_quadratic_wf3():
    x, _ = NCG(quad1, wf=3).minimize()
    assert np.allclose(quad1.jacobian(x), 0)

    x, _ = NCG(quad2, wf=3).minimize()
    assert np.allclose(quad2.jacobian(x), 0)

    x, _ = NCG(quad5, wf=3).minimize()
    assert np.allclose(quad5.jacobian(x), 0)


def test_NCG_Rosenbrock():
    obj = Rosenbrock()
    x, _ = NCG(obj, wf=3).minimize()
    assert np.allclose(x, obj.x_star)


if __name__ == "__main__":
    pytest.main()
