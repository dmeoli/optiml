import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.stochastic import AdaGrad


def test_AdaGrad_quadratic():
    assert np.allclose(AdaGrad(f=quad1).minimize().x, quad1.x_star(), rtol=0.1)
    assert np.allclose(AdaGrad(f=quad2).minimize().x, quad2.x_star(), rtol=0.1)


def test_AdaGrad_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(AdaGrad(f=rosen).minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
