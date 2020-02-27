import numpy as np
import pytest

from optimization.optimization_function import quad1, quad2, Rosenbrock
from optimization.unconstrained.accelerated_gradient import AcceleratedGradient


def test_quadratic_wf0():
    x, _ = AcceleratedGradient(quad1, wf=0, m1=0).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = AcceleratedGradient(quad2, wf=0, m1=0).minimize()
    assert np.allclose(x, quad2.x_star())

    x, _ = AcceleratedGradient(quad1, wf=0, m1=0.1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = AcceleratedGradient(quad2, wf=0, m1=0.1).minimize()
    assert np.allclose(x, quad2.x_star())


def test_quadratic_wf1():
    x, _ = AcceleratedGradient(quad1, wf=1, m1=0).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = AcceleratedGradient(quad2, wf=1, m1=0).minimize()
    assert np.allclose(x, quad2.x_star())

    x, _ = AcceleratedGradient(quad1, wf=1, m1=0.1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = AcceleratedGradient(quad2, wf=1, m1=0.1).minimize()
    assert np.allclose(x, quad2.x_star())


def test_quadratic_wf2():
    x, _ = AcceleratedGradient(quad1, wf=2, m1=0).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = AcceleratedGradient(quad2, wf=2, m1=0).minimize()
    assert np.allclose(x, quad2.x_star())

    x, _ = AcceleratedGradient(quad1, wf=2, m1=0.1).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = AcceleratedGradient(quad2, wf=2, m1=0.1).minimize()
    assert np.allclose(x, quad2.x_star())


def test_quadratic_wf3():
    x, _ = AcceleratedGradient(quad1, wf=3, m1=0).minimize()
    assert np.allclose(x, quad1.x_star())

    x, _ = AcceleratedGradient(quad2, wf=3, m1=0).minimize()
    assert np.allclose(x, quad2.x_star())

    x, _ = AcceleratedGradient(quad1, wf=3, m1=0.1).minimize()
    assert np.allclose(x, quad1.x_star(), rtol=0.1)

    x, _ = AcceleratedGradient(quad2, wf=3, m1=0.1).minimize()
    assert np.allclose(x, quad2.x_star(), rtol=0.1)


def test_Rosenbrock():
    obj = Rosenbrock()
    x, _ = AcceleratedGradient(obj, a_start=0.1).minimize()
    assert np.allclose(x, obj.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
