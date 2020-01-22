import numpy as np
import pytest

from optimization.functions import quad1, quad2, quad5, Rosenbrock
from optimization.unconstrained.accelerated_gradient import ACCG


def test_quadratic_wf0():
    x, _ = ACCG(quad1, wf=0, m1=0).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = ACCG(quad2, wf=0, m1=0).minimize()
    assert np.allclose(x, quad2.x_star)

    # x, _ = ACCG(quad5, wf=0, m1=0).minimize()
    # assert np.allclose(x, quad5.x_star, rtol=0.1)

    x, _ = ACCG(quad1, wf=0, m1=0.1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = ACCG(quad2, wf=0, m1=0.1).minimize()
    assert np.allclose(x, quad2.x_star)

    # x, _ = ACCG(quad5, wf=0, m1=0.1).minimize()
    # assert np.allclose(x, quad5.x_star, rtol=0.1)


def test_quadratic_wf1():
    x, _ = ACCG(quad1, wf=1, m1=0).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = ACCG(quad2, wf=1, m1=0).minimize()
    assert np.allclose(x, quad2.x_star)

    # x, _ = ACCG(quad5, wf=1, m1=0).minimize()
    # assert np.allclose(x, quad5.x_star, rtol=0.1)

    x, _ = ACCG(quad1, wf=1, m1=0.1).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = ACCG(quad2, wf=1, m1=0.1).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = ACCG(quad5, wf=1, m1=0.1).minimize()
    assert np.allclose(x, quad5.x_star, rtol=0.1)


def test_quadratic_wf2():
    x, _ = ACCG(quad1, wf=2, m1=0).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = ACCG(quad2, wf=2, m1=0).minimize()
    assert np.allclose(x, quad2.x_star)

    # x, _ = ACCG(quad5, wf=2, m1=0).minimize()
    # assert np.allclose(x, quad5.x_star, rtol=0.1)

    # x, _ = ACCG(quad1, wf=2, m1=0.1).minimize()
    # assert np.allclose(x, quad1.x_star)

    x, _ = ACCG(quad2, wf=2, m1=0.1).minimize()
    assert np.allclose(x, quad2.x_star)

    # x, _ = ACCG(quad5, wf=2, m1=0.1).minimize()
    # assert np.allclose(x, quad5.x_star, rtol=0.1)


def test_quadratic_wf3():
    x, _ = ACCG(quad1, wf=3, m1=0).minimize()
    assert np.allclose(x, quad1.x_star)

    x, _ = ACCG(quad2, wf=3, m1=0).minimize()
    assert np.allclose(x, quad2.x_star)

    x, _ = ACCG(quad5, wf=3, m1=0).minimize()
    assert np.allclose(x, quad5.x_star, rtol=0.1)

    # x, _ = ACCG(quad1, wf=3, m1=0.1).minimize()
    # assert np.allclose(x, quad1.x_star, rtol=0.1)

    # x, _ = ACCG(quad2, wf=3, m1=0.1).minimize()
    # assert np.allclose(x, quad2.x_star, rtol=0.1)

    # x, _ = ACCG(quad5, wf=3, m1=0.1).minimize()
    # assert np.allclose(x, quad5.x_star, rtol=0.1)


def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = ACCG(obj, a_start=0.1).minimize()
    assert np.allclose(x, obj.x_star, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
