import numpy as np
import pytest

from optiml.opti import quad1, quad2
from optiml.opti.unconstrained import Rosenbrock
from optiml.opti.unconstrained.stochastic import RProp


def test_RProp_quadratic():
    assert np.allclose(RProp(f=quad1, x=np.random.uniform(size=2)).minimize().x, quad1.x_star())
    assert np.allclose(RProp(f=quad2, x=np.random.uniform(size=2)).minimize().x, quad2.x_star())


def test_RProp_Rosenbrock():
    rosen = Rosenbrock()
    assert np.allclose(RProp(f=rosen, x=np.random.uniform(size=2)).minimize().x, rosen.x_star(), rtol=0.1)


if __name__ == "__main__":
    pytest.main()
