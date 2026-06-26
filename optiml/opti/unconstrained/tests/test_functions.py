import numpy as np
import pytest

from optiml.opti.unconstrained import Rosenbrock, Ackley, SixHumpCamel


def test_rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(rosen.x_star(), np.ones(rosen.ndim))
    assert np.isclose(rosen.f_star(), 0.)
    # the global minimum is a stationary point
    assert np.allclose(rosen.jacobian(rosen.x_star()), 0., atol=1e-6)
    # the a == 0 variant has its minimum at the origin
    assert np.allclose(Rosenbrock(a=0).x_star(), np.zeros(2))


def test_ackley():
    ackley = Ackley()
    assert np.allclose(ackley.x_star(), np.zeros(2))
    assert np.isclose(ackley.f_star(), 0., atol=1e-10)
    assert np.all(np.isfinite(ackley.jacobian(np.array([0.5, -0.3]))))


def test_six_hump_camel():
    shc = SixHumpCamel()
    # both columns of x_star are global minima with the same value
    assert np.isclose(shc.function(shc.x_star()[:, 0]),
                      shc.function(shc.x_star()[:, 1]), atol=1e-3)
    assert np.isclose(shc.f_star(), shc.function(shc.x_star()[:, 0]))
    assert np.all(np.isfinite(shc.jacobian(np.array([0.1, -0.1]))))


if __name__ == "__main__":
    pytest.main()
