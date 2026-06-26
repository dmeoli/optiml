import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.line_search import BFGS, LBFGS


def test_BFGS_quadratic():
    assert np.allclose(BFGS(f=quad1).minimize().x, quad1.x_star())
    assert np.allclose(BFGS(f=quad2).minimize().x, quad2.x_star())


def test_BFGS_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(BFGS(f=rosen).minimize().x, rosen.x_star())


def test_LBFGS_quadratic():
    assert np.allclose(LBFGS(f=quad1).minimize().x, quad1.x_star())
    assert np.allclose(LBFGS(f=quad2).minimize().x, quad2.x_star())


def test_LBFGS_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(LBFGS(f=rosen).minimize().x, rosen.x_star())


if __name__ == "__main__":
    pytest.main()
