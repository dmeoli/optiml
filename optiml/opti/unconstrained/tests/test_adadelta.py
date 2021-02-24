import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.stochastic import AdaDelta


def test_AdaDelta_quadratic():
    assert np.allclose(AdaDelta(f=quad1).minimize().x, quad1.x_star(), rtol=0.15)
    assert np.allclose(AdaDelta(f=quad2).minimize().x, quad2.x_star(), rtol=0.15)


def test_AdaDelta_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaDelta(f=rosen, step_size=0.1).minimize().x, rosen.x_star(), rtol=0.15)


if __name__ == "__main__":
    pytest.main()
